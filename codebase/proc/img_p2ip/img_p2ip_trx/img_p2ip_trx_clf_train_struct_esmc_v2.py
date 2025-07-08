import os
import sys
from pathlib import Path
path_root = Path(__file__).parents[3]
sys.path.insert(0, str(path_root))

from utils import dl_reproducible_result_util
import glob
import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.classification import AveragePrecision
from lightning.pytorch.loggers import TensorBoardLogger

from proc.img_p2ip.img_p2ip_trx_datamodule_struct_esmc_v2 import ImgP2ipCustomDataModule
from proc.img_p2ip.img_p2ip_trx.arch_CDAMViT import CDAMViT
from utils import PPIPUtils


# ################## model-dictionary declaration - start ###############################
model_dict = {'CDAMViT': CDAMViT}
# can add additional model architectures 
# ################## model-dictionary declaration - end ###############################


class ImgP2ipTrx(L.LightningModule):
    def __init__(self, root_path='./', result_dir='./', model_name='CDAMViT', config=None):
        """
        Inputs:
            model_name - name of the model/CNN to run. Used for creating the model;  
            config - Hyperparameters for the model, as dictionary.
        """
        super(ImgP2ipTrx, self).__init__()
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
        self.aupr_val = AveragePrecision(task='multiclass', num_classes=2, average='macro')
        # Model Configuration
        self.model = self.__create_model(model_name, config)
        # sample input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, config["img_resoln"], config["img_resoln"]), dtype=torch.float
        )


    def __create_model(self, model_name='CDAMViT', config=None):
        if model_name in model_dict:
            return model_dict[model_name](**config)
        else:
            assert False, f'Unknown model name "{model_name}". Available models are: {str(model_dict.keys())}'


    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        # forward function that is run when visualizing the graph
        # print('#### inside the forward() method')
        # return self.model(item.half())  # for strategy: deepspeed_stage_x
        return self.model(x)  # return logits, attn_map


    def cross_entropy_loss(self, logits, labels):
        # class_weights = torch.Tensor([100.0/90, 100.0/10]) # as class-0 entry is 90% of the entire training set
        # class_weights = class_weights.to(out)
        # loss_module = nn.CrossEntropyLoss(weight=class_weights)
        loss_module = torch.nn.CrossEntropyLoss()
        return loss_module(logits, labels)


    def training_step(self, train_batch, batch_idx):
        # REQUIRED
        # training_step defined the train loop.
        # It is independent of forward
        # print('#### inside the training_step() method')
        x, y = train_batch  # "train_batch" is the output of the training data loader.
        logits, _ = self.forward(x)  # return logits, attn_map
        loss = self.cross_entropy_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return {'loss': loss}


    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        # print('#### inside the validation_step() method')
        x, y = batch
        logits, _ = self.forward(x)  # return logits, attn_map
        loss =  self.cross_entropy_loss(logits, y)
        self.aupr_val.update(logits, y)
        self.validation_step_outputs.append({"val_loss": loss})
        return {'val_loss': loss}


    def on_validation_epoch_end(self):
        # OPTIONAL
        # called at the end of the validation epoch
        # validation_step_outputs is a list with what you returned in validation_step for each batch
        # validation_step_outputs = [{'val_loss': batch_0_loss, 'val_aupr': batch_0_aupr}, {'val_loss': batch_1_loss, 'val_aupr': batch_1_aupr}
        # , ..., {'val_loss': batch_n_loss, 'val_aupr': batch_n_aupr}] 

        # print('#### inside the validation_end() method')
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        avg_aupr = self.aupr_val.compute()
        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, sync_dist=True) # ## this is important for model checkpointing
        self.log("val_aupr", avg_aupr, on_step=False, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.clear()  # free memory
        # remember to reset metrics at the end of the epoch
        self.aupr_val.reset()
        print('on_validation_epoch_end :: val_loss= ' + str(avg_loss) + " val_aupr= " + str(avg_aupr))
        torch.cuda.empty_cache()  # ## will it work to prevent slow-down with each progressing epochs??
        return {'val_loss': avg_loss, "val_aupr": avg_aupr}


    def test_step(self, test_batch, batch_idx):
        # predict outputs for test_batch and return as dictionary
        # print('#### inside the test_step() method: batch_idx: ' + str(batch_idx))
        x, y = test_batch

        # Enable gradient tracking for CDAM generation during evaluation 
        x.requires_grad = True
        torch.set_grad_enabled(True)

        logits, attn_map = self(x)
        logits = logits.detach().cpu()
        attn_map = attn_map.detach().cpu()
        self.test_step_outputs.append({'logits': logits, 'attn': attn_map, 'y': y})
        return {'logits': logits, 'attn': attn_map, 'y': y}


    def configure_optimizers(self):
        # REQUIRED
        # print('#### inside the configure_optimizers() method')
        
        # AdamW is Adam with a correct implementation of weight decay (see here
        # for details: https://arxiv.org/pdf/1711.05101.pdf)
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.config['lr'],
            weight_decay=self.hparams.config['weight_decay']
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)  # for val_loss
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)  # for val_aupr

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss' # val_loss, val_aupr
        }
# End of ImgP2ipTrx class


def train_model(root_path='./', result_dir='./', model_name='CDAMViT', save_name='CDAMViT', config = None, prv_ckpt_path=None):
    # print('#### inside the train_model() method - Start')
    # for reproducibility set seed for pseudo-random generators in: pytorch, numpy, python.random 
    L.seed_everything(456, workers=True)
    
    # instantiate the model class
    model = ImgP2ipTrx(root_path=root_path, result_dir=result_dir, model_name=model_name, config=config)

    # Data Module
    data_module = ImgP2ipCustomDataModule(root_path=root_path, batch_size=config['batch_size'], workers=config['num_workers']
                                          , img_resoln=config['img_resoln'], spec_type=config['spec_type'], dbl_combi_flg=config['dbl_combi_flg']
                                          , weighted_sampling=config['weighted_sampling'])
    
    # define the checkpoint_callback
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
                        dirpath=os.path.join(result_dir, save_name)
                        , filename="ImgP2ipTrx" + "-epoch{epoch:02d}-val_aupr{val_aupr:.2f}-last"
                        , auto_insert_metric_name=False
                        , save_top_k=1
                        , save_last=True  # always save the last checkpoint as 'last.ckpt'
                        , monitor='val_loss', mode='min'
                        # , monitor='val_aupr', mode='max' 
                        , every_n_epochs = 1, verbose=True)
    
    # define the early_stop_callback
    early_stop_callback = L.pytorch.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss', mode='min'
        # monitor='val_aupr', mode='max' 
        , min_delta=0.00, patience=3, verbose=True)

    # define the lr_monitor_callback
    lr_monitor_callback = L.pytorch.callbacks.LearningRateMonitor("epoch")
    
    # Trainer Configuration
    trainer = L.Trainer(
        accelerator='gpu', devices=2,
        strategy='ddp',  # ddp, ddp_find_unused_parameters_true
        precision='16-mixed',
        max_epochs=config['num_epochs'],
        logger=TensorBoardLogger(save_dir=os.path.join(result_dir, 'logs')),
        callbacks=[checkpoint_callback, lr_monitor_callback],  # can also use checkpoint_callback, early_stop_callback
        enable_progress_bar = True,
        enable_model_summary = True
    )
    print('#### before calling trainer.fit(model) method')
    print('prv_ckpt_path: ' + str(prv_ckpt_path))
    if(prv_ckpt_path is None):
        print('############## Starting fresh training...')
        trainer.fit(model=model, datamodule=data_module)
    else:
        print('############## Resuming training from the previously saved checkpoint...')
        # ckpt_path = os.path.join(prv_ckpt_path, 'ImgP2ipTrx' + '*.ckpt' )
        ckpt_path = os.path.join(prv_ckpt_path, 'last.ckpt' )
        ckpt_file_name = glob.glob(ckpt_path, recursive=False)[0]

        # ################# TEMP CODE -START ############################################
        # ckpt_file_name = os.path.join(prv_ckpt_path, 'ImgP2ipTrx-epoch17-val_aupr0.90-last.ckpt' )
        # ################# TEMP CODE -END ############################################

        print('ckpt_file_name: ' + str(ckpt_file_name))
        trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_file_name)
    print('#### after calling trainer.fit(model) method')
    # print('#### inside the train_model() method - End')
# End of train_model() method


def start(root_path='./', model_name = 'CDAMViT', save_name=None, prv_ckpt_path=None):
    # print('#### inside the start() method - Start')
    if save_name is None:
        save_name = model_name

    config = {}
    if(model_name == 'CDAMViT'):
        config['img_resoln'] = 400
        config['patch_size'] = 16  # 16, 8
        config['num_heads'] = 8  # 8, 12
        config['num_layers'] = 8  # 8, 10
        config['grad_scale_lambda'] = 0.7
        config['lr'] = 3e-4
        config['weight_decay'] = 0.05
        config['batch_size'] = 24  # 24, 12
        config['num_epochs'] = 100
        config['spec_type'] = 'human'
        config['dbl_combi_flg'] = False
        config['weighted_sampling'] = True
        config["num_workers"] = 38  # 38, 4  # for the DataModule creation (in single node case)

    result_dir = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/train')
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
# End of start() method


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    # # prv_ckpt_path would be None for the fresh training
    prv_ckpt_path = None
    prv_ckpt_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/train/CDAMViT_tlStructEsmc_r400p16')
    start(root_path=root_path, model_name='CDAMViT', save_name='CDAMViT_tlStructEsmc_r400p16', prv_ckpt_path=prv_ckpt_path)
