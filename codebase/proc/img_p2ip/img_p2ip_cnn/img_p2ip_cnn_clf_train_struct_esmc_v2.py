import os
import sys
from pathlib import Path
path_root = Path(__file__).parents[3]  
sys.path.insert(0, str(path_root))
from utils import dl_reproducible_result_util
import glob
import lightning as L
import torch
import torch.nn  as nn
import torch.optim as optim
from torchmetrics.classification import AveragePrecision
from lightning.pytorch.loggers import TensorBoardLogger
from proc.img_p2ip.img_p2ip_datamodule_struct_esmc_v2 import ImgP2ipCustomDataModule
from proc.img_p2ip.img_p2ip_cnn.arch_resnet import ResNet
from utils import PPIPUtils


model_dict = {}
model_dict['ResNet'] = ResNet


class ImgP2ipCnn(L.LightningModule):
    def __init__(self, root_path='./', result_dir='./', model_name='ResNet', config=None):
        """
        Inputs:
            model_name - name of the model/CNN to run. Used for creating the model;  
            config - Hyperparameters for the model, as dictionary.
        """
        super(ImgP2ipCnn, self).__init__()
        torch.manual_seed(456)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters()
        self.root_path = root_path
        self.result_dir = result_dir
        self.model_name = model_name
        self.config = config
        self.aupr_train = AveragePrecision(task='multiclass', num_classes=2, average='macro')
        self.aupr_val = AveragePrecision(task='multiclass', num_classes=2, average='macro')
        self.model = self.__create_model(model_name, config)
        self.example_input_array = torch.zeros((1, 3, config["img_resoln"], config["img_resoln"]), dtype=torch.float)
    
    
    def __create_model(self, model_name='ResNet', config=None):
        if model_name in model_dict:
            return model_dict[model_name](**config)
        else:
            assert False, f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'
    
    
    def forward(self, item):
        return self.model(item)
    
    
    def cross_entropy_loss(self, out, labels):
        loss_module = nn.CrossEntropyLoss()
        return loss_module(out, labels)
    
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch  
        out = self.forward(x)
        loss = self.cross_entropy_loss(out, y)
        aupr = self.aupr_train(out, y)
        self.log("train/train_loss", loss)
        self.log("train/train_aupr", aupr, on_step=False, on_epoch=True, sync_dist=True)
        return {'loss': loss, 'aupr': aupr}
    
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out = self.forward(x)
        loss = self.cross_entropy_loss(out, y)
        self.aupr_val.update(out, y)
        self.validation_step_outputs.append({"val_loss": loss})
        return {"val_loss": loss}
    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        avg_aupr = self.aupr_val.compute()
        self.log("train/val_loss", avg_loss, on_step=False, on_epoch=True, sync_dist=True)  
        self.log("train/val_aupr", avg_aupr, on_step=False, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.clear()  
        self.aupr_val.reset()
        torch.cuda.empty_cache()  
        return {'val_loss': avg_loss, "val_aupr": avg_aupr}
    
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        out = self(x)
        self.test_step_outputs.append({'out': out, 'y': y})
        return {'out': out, 'y': y}
    
    
    def configure_optimizers(self):
        if self.hparams.config['optimizer_name'] == "Adam":
            optimizer = optim.AdamW(params=self.trainer.model.parameters(), lr=self.hparams.config['lr'], weight_decay=self.hparams.config['weight_decay'])
        elif self.hparams.config['optimizer_name']  == "SGD":
            optimizer = optim.SGD(params=self.trainer.model.parameters(), lr=self.hparams.config['lr'], weight_decay=self.hparams.config['weight_decay']
                                  , momentum=self.hparams.config['momentum'])
        else:
            assert False, 'Unknown optimizer:' + self.hparams.config['optimizer_name']
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.1, patience=10, threshold=0.001, threshold_mode='rel'
                                                         , cooldown=0, min_lr=0, eps=1e-08, verbose=True)  
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, 
           'monitor': 'train/val_loss'
       }


def train_model(root_path='./', result_dir='./', model_name='ResNet', save_name='ResNet', config = None, prv_ckpt_path=None):
    L.seed_everything(seed=456, workers=True)
    model = ImgP2ipCnn(root_path=root_path, result_dir=result_dir, model_name=model_name, config=config)
    data_module = ImgP2ipCustomDataModule(root_path=root_path, batch_size=config['batch_size'], workers=config['num_workers']
                                          , img_resoln=config['img_resoln'], spec_type=config['spec_type'], dbl_combi_flg=config['dbl_combi_flg']
                                          , weighted_sampling=config['weighted_sampling'])
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
                        dirpath=os.path.join(result_dir, save_name)
                        , filename="ImgP2ipCnn" + "-epoch{epoch:02d}-val_aupr{train/val_aupr:.2f}-last"
                        , auto_insert_metric_name=False
                        , save_top_k=1
                        , save_last=True  
                        , monitor='train/val_loss', mode='min'
                        , every_n_epochs = 1, verbose=True)
    early_stop_callback = L.pytorch.callbacks.early_stopping.EarlyStopping(
        monitor="train/val_loss"
        , mode="min"
        , min_delta=0.00, patience=3, verbose=True)  
    lr_monitor_callback = L.pytorch.callbacks.LearningRateMonitor("epoch")
    tb_dir = os.path.join(result_dir, save_name, 'tb_logs')
    PPIPUtils.makeDir(tb_dir)
    logger = TensorBoardLogger(save_dir=tb_dir, name='', version=None, prefix=''
                               , log_graph=True, default_hp_metric=False)
    trainer = L.Trainer( default_root_dir=os.path.join(result_dir, save_name)  
                        , max_epochs=config['num_epochs']  
                        , deterministic='warn'
                        , logger=logger
                        , callbacks=[checkpoint_callback, lr_monitor_callback]  
                        , accelerator="gpu", devices=2, num_nodes=2  
                        , strategy='ddp'  
                        , precision = '16-mixed'
                        , profiler="simple"  
                        , enable_progress_bar = True
                        , enable_model_summary = True)
    if(prv_ckpt_path is None):
        trainer.fit(model=model, datamodule=data_module)
    else:
        ckpt_path = os.path.join(prv_ckpt_path, 'last.ckpt' )
        ckpt_file_name = glob.glob(ckpt_path, recursive=False)[0]
        trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_file_name)


def start(root_path='./', model_name = 'ResNet', save_name=None, prv_ckpt_path=None):
    if save_name is None:
        save_name = model_name
    config = {}
    if(model_name == 'ResNet'):
        config["num_classes"] = 2  
        config["c_hidden"] = [16, 32, 64, 128, 256]  
        config["num_blocks"] = [3, 3, 4, 5, 3]  
        config["act_fn_name"] = 'relu'  
        config["block_name"] = 'ResNetBlock'  
        config["optimizer_name"] = 'SGD'  
        config["lr"] = 0.1  
        config["momentum"] = 0.9  
        config["weight_decay"] = 1e-4  
        config["batch_size"] =  48
        config["img_resoln"] = 400  
        config["dbl_combi_flg"] = True  
        config["spec_type"] = 'human'
        config["weighted_sampling"] = True
        config["num_epochs"] = 100
        config["num_workers"] = 35  
    result_dir = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/train')
    try:
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    except OSError as ex:
        errorMessage = "Creation of the directory " + result_dir + " failed. Exception is: " + str(ex)
        raise Exception(errorMessage)
    else:
        print("Successfully created the directory %s " % result_dir)

    train_model(root_path=root_path, result_dir=result_dir, model_name=model_name, save_name=save_name, config=config, prv_ckpt_path=prv_ckpt_path)



if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    prv_ckpt_path = None
    prv_ckpt_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/train/ResNet_tlStructEsmc_r400n18D')
    start(root_path=root_path, model_name='ResNet', save_name='ResNet_tlStructEsmc_r400n18D', prv_ckpt_path=prv_ckpt_path)
