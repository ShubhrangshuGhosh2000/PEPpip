import os
import sys
from pathlib import Path
import ray.train
path_root = Path(__file__).parents[3]
sys.path.insert(0, str(path_root))
from utils import dl_reproducible_result_util
import glob
import joblib
import lightning as L
import torch
import torch.nn  as nn
import torch.optim as optim
from torchmetrics.classification import AveragePrecision
from lightning.pytorch.loggers import TensorBoardLogger
import ray
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback, TuneReportCheckpointCallback)
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining 
from codebase.proc.img_p2ip.img_p2ip_datamodule import ImgP2ipCustomDataModule
from proc.img_p2ip.img_p2ip_trx.arch_CDAMViT import CDAMViT


class _TuneReportCheckpointCallback(TuneReportCheckpointCallback, L.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


model_dict = {}
model_dict['CDAMViT'] = CDAMViT

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
        self.aupr_train = AveragePrecision(task='multiclass', num_classes=2, average='macro')
        self.aupr_val = AveragePrecision(task='multiclass', num_classes=2, average='macro')
        self.config["c_hidden"] = [16, 32, 64, 128, 256, 512]  
        self.config["num_blocks"] = [3, 3, 4, 8, 12, 4]  
        self.model = self.__create_model(model_name, config)
        self.example_input_array = torch.zeros((1, 3, config["img_resoln"], config["img_resoln"]), dtype=torch.float)
    
    
    def __create_model(self, model_name='CDAMViT', config=None):
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
        logits, _ = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        aupr = self.aupr_train(logits, y)
        self.log("train/train_loss", loss)
        self.log("train/train_aupr", aupr, on_step=False, on_epoch=True, sync_dist=True)
        return {'loss': loss, 'aupr': aupr}
    
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits, _ = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.aupr_val.update(logits, y)
        self.validation_step_outputs.append({"val_loss": loss})
        return {"val_loss": loss}
    
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        avg_aupr = self.aupr_val.compute()
        self.log("train/val_loss", avg_loss, on_step=False, on_epoch=True, sync_dist=True)  
        self.log("train/val_aupr", avg_aupr, on_step=False, on_epoch=True, sync_dist=True)
        self.validation_step_outputs.clear()  
        self.aupr_val.reset()
        return {'val_loss': avg_loss, "val_aupr": avg_aupr}
    
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits, attn_map = self(x)
        self.test_step_outputs.append({'logits': logits, 'attn': attn_map, 'y': y})
        return {'logits': logits, 'attn': attn_map, 'y': y}
    
    
    def configure_optimizers(self):
        if self.hparams.config['optimizer_name'] == "Adam":
            optimizer = optim.AdamW(params=self.trainer.model.parameters(), lr=self.hparams.config['lr'], weight_decay=self.hparams.config['weight_decay'])
        elif self.hparams.config['optimizer_name']  == "SGD":
            optimizer = optim.SGD(params=self.trainer.model.parameters(), lr=self.hparams.config['lr'], weight_decay=self.hparams.config['weight_decay']
                                  , momentum=self.hparams.config['momentum'])
        else:
            assert False, 'Unknown optimizer:' + self.hparams.config['optimizer_name']
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.1, patience=3, threshold=0.0001, threshold_mode='rel'
                                                         , cooldown=0, min_lr=0, eps=1e-08, verbose=True)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, 
           'monitor': 'train/val_loss'
       }


def train_model(config = None, root_path='./', result_dir='./', model_name='CDAMViT', save_name='CDAMViT', prv_ckpt_path=None):
    L.seed_everything(seed=456, workers=True)
    model = ImgP2ipTrx(root_path=root_path, result_dir=result_dir, model_name=model_name, config=config)
    data_module = ImgP2ipCustomDataModule(root_path=root_path, batch_size=config['batch_size'], workers=config['num_workers']
                                          , img_resoln=config['img_resoln'], spec_type=config['spec_type'], dbl_combi_flg=config['dbl_combi_flg']
                                          , weighted_sampling=config['weighted_sampling'])
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
                        dirpath=os.path.join(result_dir, save_name)
                        , filename="ImgP2ipTrx" + "-epoch{epoch:02d}-val_aupr{train/val_aupr:.2f}-last"
                        , auto_insert_metric_name=False
                        , save_top_k=1
                        , monitor='train/val_loss', mode='min'
                        , every_n_epochs = 1, verbose=True)
    early_stop_callback = L.pytorch.callbacks.early_stopping.EarlyStopping(
        monitor="train/val_loss"
        , mode="min"
        , min_delta=0.00, patience=7, verbose=True)  
    lr_monitor_callback = L.pytorch.callbacks.LearningRateMonitor("epoch")
    tune_report_chkpt_callback = _TuneReportCheckpointCallback(
                            metrics={
                                "val_loss": "train/val_loss",
                                "val_aupr": "train/val_aupr"
                            },
                            filename=model_name + "_checkpoint.ckpt",
                            on="validation_end")
    tb_dir = ray.train.get_context().get_trial_dir()
    logger = TensorBoardLogger(save_dir=tb_dir, name='', version=None, prefix=''
                               , log_graph=False, default_hp_metric=False)
    trainer = L.Trainer( default_root_dir=os.path.join(result_dir, save_name)  
                        , max_epochs=config['num_epochs']  
                        , deterministic='warn'
                        , logger=logger
                        , callbacks=[tune_report_chkpt_callback, lr_monitor_callback, early_stop_callback]  
                        , accelerator="gpu", devices=1, num_nodes=1
                        , precision = '16-mixed'
                        , enable_progress_bar = True
                        , enable_model_summary = True)
    if(prv_ckpt_path is None):
        trainer.fit(model=model, datamodule=data_module)
    else:
        ckpt_path = os.path.join(prv_ckpt_path, 'ImgP2ipTrx' + '*.ckpt' )
        ckpt_file_name = glob.glob(ckpt_path, recursive=False)[0]
        trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_file_name)


def ray_hparam_tuning(config = None, root_path='./', result_dir='./', model_name='CDAMViT', cpus_per_trial=15, gpus_per_trial=1, num_samples=10, scheduler_name='ASHA'):
    scheduler = None
    if(scheduler_name == 'ASHA'):
        scheduler = ASHAScheduler(
            max_t=config["num_epochs"] ,
            grace_period=1,
            reduction_factor=2)
    elif(scheduler_name == 'PBT'):
        scheduler = PopulationBasedTraining(
            perturbation_interval=4,
            hyperparam_mutations={
                "lr": ray.tune.loguniform(1e-4, 1e-1)
                ,"batch_size": [32, 64, 128]
            })
    reporter = CLIReporter(
        parameter_columns=["optimizer_name", "lr", "batch_size", "dbl_combi_flg"],
        metric_columns=["val_loss", "val_aupr", "training_iteration"],
        max_report_frequency=30)
    train_fn_with_parameters = ray.tune.with_parameters(train_model
                                                    , root_path=root_path
                                                    , result_dir=result_dir
                                                    , model_name=model_name
                                                    )
    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}
    sync_config=ray.train.SyncConfig()
    analysis = ray.tune.run(
        run_or_experiment=train_fn_with_parameters,  
        storage_path=result_dir,
        resources_per_trial=resources_per_trial,
        metric="val_aupr", mode="max",
        config=config,
        num_samples=num_samples,  
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_" + str(scheduler_name) + "_CDAMViT",
        resume ="AUTO",  
        reuse_actors=False,  
        log_to_file=True,  
        keep_checkpoints_num=1,  
        checkpoint_score_attr='max-val_aupr',  
        sync_config = sync_config,
        max_failures = 1,  
        raise_on_failed_trial = False,  
        verbose = 3)
    best_trial = analysis.best_trial  
    best_config = analysis.best_config  
    experiment_path = analysis.experiment_path  
    best_checkpoint = analysis.best_checkpoint  
    best_result = analysis.best_result  
    best_result_df = analysis.best_result_df  
    best_result_df['num_epochs'] = [config["num_epochs"]]
    best_result_df_name_with_path = os.path.join(result_dir, 'best_result_df.csv')
    best_result_df.to_csv(best_result_df_name_with_path, index=False)
    best_result['num_epochs'] = config["num_epochs"]
    best_result['best_config'] = best_config
    best_result['best_checkpoint_path'] = str(analysis.best_checkpoint)
    best_result_artifact_name_with_path = os.path.join(result_dir, 'best_result_dict.pkl')
    joblib.dump(value=best_result,
                filename=best_result_artifact_name_with_path,
                compress=3)


def start(root_path='./', model_name = 'CDAMViT'):
    ray.init(runtime_env={"py_modules": [os.path.join(path_root,'proc')]})
    scheduler_name = 'ASHA'  
    config = {}
    if(model_name == 'CDAMViT'):
        config["patch_size"] = ray.tune.grid_search([16])  
        config["num_heads"] = ray.tune.choice([8, 12])
        config["hidden_dim"] = ray.tune.choice([768])  
        config["batch_size"] =  ray.tune.grid_search([24])  
        config["num_classes"] = 2  
        config["optimizer_name"] = ray.tune.grid_search(['Adam'])  
        config["momentum"] = 0.9  
        config["weight_decay"] = 1e-4  
        config["img_resoln"] = 400  
        config["dbl_combi_flg"] = ray.tune.grid_search([False, True])  
        config["spec_type"] = 'human'
        config["weighted_sampling"] = ray.tune.grid_search([False, True]) 
        config["num_epochs"] = 15
        config["num_workers"] = 2  
    if(scheduler_name == 'ASHA'):
        config["lr"] = ray.tune.loguniform(1e-5, 1e-3)  
    elif(scheduler_name == 'PBT'):
        config["lr"] = 1e-2  
    cpus_per_trial = 8  
    gpus_per_trial = 1 
    num_samples = 1  
    result_dir = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_trx/train/CDAMViT_tune_results')
    try:
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    except OSError as ex:
        errorMessage = "Creation of the directory " + result_dir + " failed. Exception is: " + str(ex)
        raise Exception(errorMessage)
    else:
        print("Successfully created the directory %s " % result_dir)

    ray_hparam_tuning(config=config, root_path=root_path, result_dir=result_dir, model_name=model_name
                      , cpus_per_trial=cpus_per_trial, gpus_per_trial=gpus_per_trial
                      , num_samples=num_samples, scheduler_name=scheduler_name)


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    start(root_path=root_path, model_name='CDAMViT')
