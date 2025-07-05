import os
import sys
from pathlib import Path

path_root = Path(__file__).parents[3]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
import glob
import lightning as L
import torch
import torch.nn  as nn
import torch.optim as optim
from torchmetrics.classification import AveragePrecision
from lightning.pytorch.loggers import TensorBoardLogger

from proc.img_p2ip.img_p2ip_datamodule import ImgP2ipCustomDataModule
from proc.img_p2ip.img_p2ip_cnn.arch_resnet import ResNet
from utils import PPIPUtils


# ################## model-dictionary declaration - start ###############################
model_dict = {}
model_dict['ResNet'] = ResNet
# can add additional model architectures like DenseNet, GoogleNet etc.
# model_dict['DenseNet'] = DenseNet
# ################## model-dictionary declaration - end ###############################


class ImgP2ipCnn(L.LightningModule):
    def __init__(self, root_path='./', result_dir='./', model_name='ResNet', config=None):
        """
        Inputs:
            model_name - name of the model/CNN to run. Used for creating the model;  
            config - Hyperparameters for the model, as dictionary.
        """
        super(ImgP2ipCnn, self).__init__()
        torch.manual_seed(456)

        # instance variable declaration for collecting all the outputs from each validation_step() method 
        self.validation_step_outputs = []
        # instance variable declaration for collecting all the outputs from each test_step() method 
        self.test_step_outputs = []

        # save __init__ arguments to "self.hparams" namespace
        self.save_hyperparameters()

        self.root_path = root_path
        self.result_dir = result_dir
        self.model_name = model_name
        self.config = config

        # define performance metrics 
        self.aupr_train = AveragePrecision(task='multiclass', num_classes=2, average='macro')
        self.aupr_val = AveragePrecision(task='multiclass', num_classes=2, average='macro')

        # define the model architecture
        self.model = self.__create_model(model_name, config)
        # sample input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, config["img_resoln"], config["img_resoln"]), dtype=torch.float)


    def __create_model(self, model_name='ResNet', config=None):
        if model_name in model_dict:
            return model_dict[model_name](**config)
        else:
            assert False, f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'


    def forward(self, item):
        # in lightning, forward defines the prediction/inference actions
        # forward function that is run when visualizing the graph
        # print('#### inside the forward() method')
        # return self.model(item.half())  # for strategy: deepspeed_stage_x
        return self.model(item)


    def cross_entropy_loss(self, out, labels):
        # class_weights = torch.Tensor([100.0/90, 100.0/10]) # as class-0 entry is 90% of the entire training set
        # class_weights = class_weights.to(out)
        # loss_module = nn.CrossEntropyLoss(weight=class_weights)
        loss_module = nn.CrossEntropyLoss()
        return loss_module(out, labels)


    # def accuracy(self, out, labels):
    #     # _, predicted = torch.max(out.data, 1)
    #     predicted = out.argmax(dim=-1)
    #     acc = (predicted == labels).float().mean()
    #     return acc


    def training_step(self, train_batch, batch_idx):
        # REQUIRED
        # training_step defined the train loop.
        # It is independent of forward
        # print('#### inside the training_step() method')
        x, y = train_batch  # "train_batch" is the output of the training data loader.
        out = self.forward(x)
        loss = self.cross_entropy_loss(out, y)
        aupr = self.aupr_train(out, y)
        self.log("train/train_loss", loss)
        self.log("train/train_aupr", aupr, on_step=False, on_epoch=True, sync_dist=True)
        return {'loss': loss, 'aupr': aupr}


    def validation_step(self, val_batch, batch_idx):
        # OPTIONAL
        # print('#### inside the validation_step() method')
        x, y = val_batch
        out = self.forward(x)
        loss = self.cross_entropy_loss(out, y)
        self.aupr_val.update(out, y)
        self.validation_step_outputs.append({"val_loss": loss})
        return {"val_loss": loss}


    def on_validation_epoch_end(self):
        # OPTIONAL
        # called at the end of the validation epoch
        # validation_step_outputs is a list with what you returned in validation_step for each batch
        # validation_step_outputs = [{'val_loss': batch_0_loss, 'val_aupr': batch_0_aupr}, {'val_loss': batch_1_loss, 'val_aupr': batch_1_aupr}
        # , ..., {'val_loss': batch_n_loss, 'val_aupr': batch_n_aupr}] 

        # print('#### inside the validation_end() method')
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        avg_aupr = self.aupr_val.compute()
        self.log("train/val_loss", avg_loss, on_step=False, on_epoch=True, sync_dist=True)  # ## this is important for model checkpointing
        self.log("train/val_aupr", avg_aupr, on_step=False, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory
        # remember to reset metrics at the end of the epoch
        self.aupr_val.reset()
        print('on_validation_epoch_end :: val_loss= ' + str(avg_loss) + " val_aupr= " + str(avg_aupr))
        return {'val_loss': avg_loss, "val_aupr": avg_aupr}


    def test_step(self, test_batch, batch_idx):
        # predict outputs for test_batch and return as dictionary
        # print('#### inside the test_step() method: batch_idx: ' + str(batch_idx))
        x, y = test_batch
        out = self(x)
        self.test_step_outputs.append({'out': out, 'y': y})
        return {'out': out, 'y': y}


    def configure_optimizers(self):
        # REQUIRED
        # print('#### inside the configure_optimizers() method')
        # We will support Adam or SGD as optimizers.
        if self.hparams.config['optimizer_name'] == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(params=self.trainer.model.parameters(), lr=self.hparams.config['lr'], weight_decay=self.hparams.config['weight_decay'])
        elif self.hparams.config['optimizer_name']  == "SGD":
            optimizer = optim.SGD(params=self.trainer.model.parameters(), lr=self.hparams.config['lr'], weight_decay=self.hparams.config['weight_decay']
                                  , momentum=self.hparams.config['momentum'])
        else:
            assert False, 'Unknown optimizer:' + self.hparams.config['optimizer_name']

        # MultiStepLR scheduler configuration
        # We will reduce the learning rate by 0.1 after 50, 100 and 150 epochs
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20], gamma=0.1)

        # ReduceLROnPlateau scheduler configuration
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.1, patience=3, threshold=0.0001, threshold_mode='rel'
                                                         , cooldown=0, min_lr=0, eps=1e-08, verbose=True)
        # return [optimizer], [scheduler]
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, # Changed scheduler to lr_scheduler
           'monitor': 'train/val_loss'
       }


def train_model(root_path='./', result_dir='./', model_name='ResNet', save_name='ResNet', config = None, prv_ckpt_path=None):
    print('#### inside the train_model() method - Start')
    # for reproducibility set seed for pseudo-random generators in: pytorch, numpy, python.random 
    L.seed_everything(seed=456, workers=True)

    # instantiate the model class
    model = ImgP2ipCnn(root_path=root_path, result_dir=result_dir, model_name=model_name, config=config)
    data_module = ImgP2ipCustomDataModule(root_path=root_path, batch_size=config['batch_size'], workers=config['num_workers']
                                          , img_resoln=config['img_resoln'], spec_type=config['spec_type'], dbl_combi_flg=config['dbl_combi_flg']
                                          , weighted_sampling=config['weighted_sampling'])

    # define the checkpoint_callback
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
                        dirpath=os.path.join(result_dir, save_name)
                        , filename="ImgP2ipCnn" + "-epoch{epoch:02d}-val_aupr{train/val_aupr:.2f}-last"
                        , auto_insert_metric_name=False
                        , save_top_k=1
                        , monitor='train/val_loss', mode='min'
                        # , monitor='train/val_aupr', mode='max'
                        , every_n_epochs = 1, verbose=True)

     # define the early_stop_callback
    early_stop_callback = L.pytorch.callbacks.early_stopping.EarlyStopping(
        monitor="train/val_loss"
        # monitor="train/val_aupr"
        , mode="min"
        , min_delta=0.00, patience=3, verbose=True)  # monitor="val_aupr" can be checked

    # define the lr_monitor_callback
    lr_monitor_callback = L.pytorch.callbacks.LearningRateMonitor("epoch")

    # instantiate tensorboard logger
    tb_dir = os.path.join(result_dir, save_name, 'tb_logs')
    PPIPUtils.makeDir(tb_dir)
    logger = TensorBoardLogger(save_dir=tb_dir, name='', version=None, prefix=''
                               , log_graph=True, default_hp_metric=False)

    # instantiate a trainer
    trainer = L.Trainer( default_root_dir=os.path.join(result_dir, save_name)  # Default path for logs and weights when no logger/ckpt_callback passed
                        , max_epochs=config['num_epochs']  # How many epochs to train for if no patience is set
                        , deterministic='warn'
                        , logger=logger
                        # , logger = False
                        , callbacks=[checkpoint_callback, lr_monitor_callback]  # can also use checkpoint_callback, early_stop_callback
                        , accelerator="gpu", devices=2, num_nodes=1  # for multi-gpu in a single node training
                        # , accelerator="gpu", devices=2, num_nodes=2  # for multi-gpu, multi-node training
                        , strategy='ddp'  # ddp, fsdp, deepspeed_stage_2
                        , precision = '16-mixed'
                        , enable_progress_bar = True
                        , enable_model_summary = True)
    print('#### before calling trainer.fit(model) method')
    print('prv_ckpt_path: ' + str(prv_ckpt_path))
    if(prv_ckpt_path is None):
        print('############## Starting fresh training...')
        trainer.fit(model=model, datamodule=data_module)
    else:
        print('############## Resuming training from the previously saved checkpoint...')
        ckpt_path = os.path.join(prv_ckpt_path, 'ImgP2ipCnn' + '*.ckpt' )
        ckpt_file_name = glob.glob(ckpt_path, recursive=False)[0]
        print('ckpt_file_name: ' + str(ckpt_file_name))
        trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_file_name)
    print('#### after calling trainer.fit(model) method')
    # print('#### inside the train_model() method - End')


def start(root_path='./', model_name = 'ResNet', save_name=None, prv_ckpt_path=None):
    # print('#### inside the start() method - Start')
    if save_name is None:
        save_name = model_name

    config = {}
    if(model_name == 'ResNet'):
        config["num_classes"] = 2  # hparam for the model architecture
        # ## for ResNet9
        # config["c_hidden"] = [16, 32, 64]  # hparam for the model architecture
        # config["num_blocks"] = [3, 3, 3]  # hparam for the model architecture

        # ## for ResNet18
        # config["c_hidden"] = [16, 32, 64, 128, 256]  # hparam for the model architecture
        # config["num_blocks"] = [3, 3, 4, 5, 3]  # hparam for the model architecture

        # ## for ResNet22
        config["c_hidden"] = [16, 32, 64, 128, 256]  # hparam for the model architecture
        config["num_blocks"] = [3, 4, 5, 6, 4]  # hparam for the model architecture

        # ## for ResNet34
        # config["c_hidden"] = [16, 32, 64, 128, 256, 512]  # hparam for the model architecture
        # config["num_blocks"] = [3, 3, 4, 8, 12, 4]  # hparam for the model architecture

        config["act_fn_name"] = 'relu'  # tanh, relu, leakyrelu, gelu
        config["block_name"] = 'ResNetBlock'  # ResNetBlock, PreActResNetBlock
        config["optimizer_name"] = 'SGD'  # Adam, SGD (FOR RESNET BOTH NEEDS TO BE CHECKED)
        config["lr"] = 0.1  # For SGD: 0.1; Adam: 1e-3
        config["momentum"] = 0.9  # hparam for the SGD optimizer
        config["weight_decay"] = 1e-4  # For SGD: 1e-4; Adam: 1e-4
        config["batch_size"] =  20
        config["img_resoln"] = 800  # img_resoln can be 256 or 400 or 800
        config["dbl_combi_flg"] = True  # flag to indicate whether (prot1, prot2) pair and (prot2, prot1) pair both will be considered in the training data (as they will give rise to 2 different figures)
        config["spec_type"] = 'human'
        # 'weighted_sampling' flag is used to indicate whether WeightedRandomSampler would be used 
        # for the batch selection in the DataLoader. It is useful for the trainiing of the dataset with the skewed class distribution.
        config["weighted_sampling"] = True
        config["num_epochs"] = 20
        config["num_workers"] = os.cpu_count() - 5  # for the DataModule creation

    result_dir = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/train')
    try:
        # check if the result_dir already exists and if not, then create it
        if not os.path.exists(result_dir):
            print("The directory: " + result_dir + " does not exist.. Creating it...")
            os.makedirs(result_dir)
    except OSError as ex:
        errorMessage = "Creation of the directory " + result_dir + " failed. Exception is: " + str(ex)
        raise Exception(errorMessage)
    else:
        print("Successfully created the directory %s " % result_dir)

    train_model(root_path=root_path, result_dir=result_dir, model_name=model_name, save_name=save_name, config=config, prv_ckpt_path=prv_ckpt_path)

    # perform the final training
    # print('\n# ############################# performing the final training #############################\n')
    # img_p2ip_cnn_clf_train_final.start(root_path)

    # perform the testing 
    # print('\n# ############################# performing the testing #############################\n')
    # img_p2ip_cnn_clf_test.start(root_path = root_path)
    
    print('\n ##################### END OF THE ENTIRE PROCESS ######################')


if __name__ == '__main__':
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/mat_p2ip_prj_working/')
    # # prv_ckpt_path would be None for the fresh training
    prv_ckpt_path = None
    # prv_ckpt_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/train/ResNet_expt_res256_rn9_NoLossWeight_dbl')
    start(root_path=root_path, model_name='ResNet', save_name='ResNet_expt_r800n22D', prv_ckpt_path=prv_ckpt_path)
