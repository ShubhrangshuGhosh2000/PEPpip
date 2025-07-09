import os
import sys
from pathlib import Path
path_root = Path(__file__).parents[3]
sys.path.insert(0, str(path_root))
from utils import dl_reproducible_result_util
import glob
import lightning as L
import torch
from torchmetrics.classification import AveragePrecision
from lightning.pytorch.loggers import TensorBoardLogger
from proc.img_p2ip.img_p2ip_trx_datamodule_struct_esmc_v2 import ImgP2ipCustomDataModule
from proc.img_p2ip.img_p2ip_trx.arch_CDAMViT import CDAMViT


model_dict = {'CDAMViT': CDAMViT}

class ImgP2ipTrx(L.LightningModule):
    def __init__(self, root_path='./', result_dir='./', model_name='CDAMViT', config=None):
        """
        Inputs:
            model_name - name of the model/CNN to run. Used for creating the model;  
            config - Hyperparameters for the model, as dictionary.
        """
        super(ImgP2ipTrx, self).__init__()
        torch.manual_seed(456)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters()
        self.root_path = root_path
        self.result_dir = result_dir
        self.model_name = model_name
        self.config = config
        self.aupr_val = AveragePrecision(task='multiclass', num_classes=2, average='macro')
        self.model = self.__create_model(model_name, config)
        self.example_input_array = torch.zeros((1, 3, config["img_resoln"], config["img_resoln"]), dtype=torch.float
        )
    
    
    def __create_model(self, model_name='CDAMViT', config=None):
        if model_name in model_dict:
            return model_dict[model_name](**config)
        else:
            assert False, f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'
    
    
    def forward(self, x):
        return self.model(x)  
    
    
    def cross_entropy_loss(self, logits, labels):
        loss_module = torch.nn.CrossEntropyLoss()
        return loss_module(logits, labels)
    
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch  
        logits, _ = self.forward(x)  
        loss = self.cross_entropy_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return {'loss': loss}
    
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self.forward(x)  
        loss =  self.cross_entropy_loss(logits, y)
        self.aupr_val.update(logits, y)
        self.validation_step_outputs.append({"val_loss": loss})
        return {'val_loss': loss}
    
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        avg_aupr = self.aupr_val.compute()
        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, sync_dist=True) 
        self.log("val_aupr", avg_aupr, on_step=False, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.clear()  
        self.aupr_val.reset()
        torch.cuda.empty_cache()  
        return {'val_loss': avg_loss, "val_aupr": avg_aupr}
    
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x.requires_grad = True
        torch.set_grad_enabled(True)
        logits, attn_map = self(x)
        logits = logits.detach().cpu()
        attn_map = attn_map.detach().cpu()
        self.test_step_outputs.append({'logits': logits, 'attn': attn_map, 'y': y})
        return {'logits': logits, 'attn': attn_map, 'y': y}
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.config['lr'],
            weight_decay=self.hparams.config['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)  
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss' 
        }


def train_model(root_path='./', result_dir='./', model_name='CDAMViT', save_name='CDAMViT', config = None, prv_ckpt_path=None):
    L.seed_everything(456, workers=True)
    model = ImgP2ipTrx(root_path=root_path, result_dir=result_dir, model_name=model_name, config=config)
    data_module = ImgP2ipCustomDataModule(root_path=root_path, batch_size=config['batch_size'], workers=config['num_workers']
                                          , img_resoln=config['img_resoln'], spec_type=config['spec_type'], dbl_combi_flg=config['dbl_combi_flg']
                                          , weighted_sampling=config['weighted_sampling'])
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
                        dirpath=os.path.join(result_dir, save_name)
                        , filename="ImgP2ipTrx" + "-epoch{epoch:02d}-val_aupr{val_aupr:.2f}-last"
                        , auto_insert_metric_name=False
                        , save_top_k=1
                        , save_last=True  
                        , monitor='val_loss', mode='min'
                        , every_n_epochs = 1, verbose=True)
    early_stop_callback = L.pytorch.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss', mode='min'
        , min_delta=0.00, patience=3, verbose=True)
    lr_monitor_callback = L.pytorch.callbacks.LearningRateMonitor("epoch")
    trainer = L.Trainer(
        accelerator='gpu', devices=2,
        strategy='ddp',  
        precision='16-mixed',
        max_epochs=config['num_epochs'],
        logger=TensorBoardLogger(save_dir=os.path.join(result_dir, 'logs')),
        callbacks=[checkpoint_callback, lr_monitor_callback],  
        enable_progress_bar = True,
        enable_model_summary = True
    )
    if(prv_ckpt_path is None):
        trainer.fit(model=model, datamodule=data_module)
    else:
        ckpt_path = os.path.join(prv_ckpt_path, 'last.ckpt' )
        ckpt_file_name = glob.glob(ckpt_path, recursive=False)[0]
        trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_file_name)


def start(root_path='./', model_name = 'CDAMViT', save_name=None, prv_ckpt_path=None):
    if save_name is None:
        save_name = model_name
    config = {}
    if(model_name == 'CDAMViT'):
        config['img_resoln'] = 400
        config['patch_size'] = 16  
        config['num_heads'] = 8  
        config['num_layers'] = 8  
        config['grad_scale_lambda'] = 0.7
        config['lr'] = 3e-4
        config['weight_decay'] = 0.05
        config['batch_size'] = 24  
        config['num_epochs'] = 100
        config['spec_type'] = 'human'
        config['dbl_combi_flg'] = False
        config['weighted_sampling'] = True
        config["num_workers"] = 38  
    result_dir = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/train')
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
    prv_ckpt_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/train/CDAMViT_tlStructEsmc_r400p16')
    start(root_path=root_path, model_name='CDAMViT', save_name='CDAMViT_tlStructEsmc_r400p16', prv_ckpt_path=prv_ckpt_path)
