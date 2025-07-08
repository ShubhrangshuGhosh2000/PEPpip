import os
import sys
from pathlib import Path

import ray.train

path_root = Path(__file__).parents[3]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

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
from ray.air import CheckpointConfig

from proc.img_p2ip.img_p2ip_datamodule_struct_esmc_v2 import ImgP2ipCustomDataModule
from proc.img_p2ip.img_p2ip_cnn.arch_resnet import ResNet
from utils import PPIPUtils


# #### This is a workaround for lightning.pytorch (latest) vs. pytorch_lightning (older) package issue -Start
# see https://github.com/ray-project/ray/issues/33426
class _TuneReportCheckpointCallback(TuneReportCheckpointCallback, L.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
# #### This is a workaround for lightning.pytorch (latest) vs. pytorch_lightning (older) package issue -End


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
        print('**** resNet_arch_type: ' + str(self.config["resNet_arch_type"]))
        if(self.config["resNet_arch_type"] == 9):
            # ## for ResNet9
            self.config["c_hidden"] = [16, 32, 64]  # hparam for the model architecture
            self.config["num_blocks"] = [3, 3, 3]  # hparam for the model architecture
        elif(self.config["resNet_arch_type"] == 18):
            # ## for ResNet18
            self.config["c_hidden"] = [16, 32, 64, 128, 256]  # hparam for the model architecture
            self.config["num_blocks"] = [3, 3, 4, 5, 3]  # hparam for the model architecture
        elif(self.config["resNet_arch_type"] == 22):
            # ## for ResNet22
            self.config["c_hidden"] = [16, 32, 64, 128, 256]  # hparam for the model architecture
            self.config["num_blocks"] = [3, 4, 5, 6, 4]  # hparam for the model architecture
        elif(self.config["resNet_arch_type"] == 34):
            ## for ResNet34
            self.config["c_hidden"] = [16, 32, 64, 128, 256, 512]  # hparam for the model architecture
            self.config["num_blocks"] = [3, 3, 4, 8, 12, 4]  # hparam for the model architecture

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
            optimizer = optim.AdamW(params=self.trainer.model.parameters(), lr=self.hparams.config['lr'][0], weight_decay=self.hparams.config['weight_decay'])
        elif self.hparams.config['optimizer_name']  == "SGD":
            optimizer = optim.SGD(params=self.trainer.model.parameters(), lr=self.hparams.config['lr'][0], weight_decay=self.hparams.config['weight_decay']
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


def train_model(config = None, root_path='./', result_dir='./', model_name='ResNet', save_name='ResNet', prv_ckpt_path=None):
    print('#### inside the train_model() method - Start')
    print('root_path: ' + str(root_path))
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
        , min_delta=0.00, patience=7, verbose=True)  # monitor="val_aupr" can be checked

    # define the lr_monitor_callback
    lr_monitor_callback = L.pytorch.callbacks.LearningRateMonitor("epoch")

    # This callback will take the val_loss and val_aupr values from the PyTorch Lightning trainer and
    # report them to Tune as the loss and mean_aupr, respectively.
    # Since Tune requires a call to tune.report() after creating a new checkpoint to register it, we
    # will use a combined reporting and checkpointing callback
    tune_report_chkpt_callback = _TuneReportCheckpointCallback(
                            metrics={
                                "val_loss": "train/val_loss",
                                "val_aupr": "train/val_aupr"
                            },
                            filename=model_name + "_checkpoint.ckpt",
                            on="validation_end")

    # instantiate tensconorboard logger
    # tb_dir = os.path.join(result_dir, save_name, 'tb_logs')
    # PPIPUtils.makeDir(tb_dir)
    tb_dir = ray.train.get_context().get_trial_dir()
    logger = TensorBoardLogger(save_dir=tb_dir, name='', version=None, prefix=''
                               , log_graph=True, default_hp_metric=False)

    # instantiate a trainer
    trainer = L.Trainer( default_root_dir=os.path.join(result_dir, save_name)  # Default path for logs and weights when no logger/ckpt_callback passed
                        , max_epochs=config['num_epochs']  # How many epochs to train for if no patience is set
                        , deterministic='warn'
                        , logger=logger
                        # , logger = False
                        , callbacks=[tune_report_chkpt_callback, lr_monitor_callback, early_stop_callback]  # can also use checkpoint_callback, early_stop_callback
                        , accelerator="gpu", devices=1, num_nodes=1
                        # ##, accelerator="gpu", devices=2, num_nodes=1  # for multi-gpu in a single node training
                        # ##, accelerator="gpu", devices=2, num_nodes=2  # for multi-gpu, multi-node training
                        # ## , strategy='ddp'  # ddp, fsdp, deepspeed_stage_2 ## DO NOT use this argument while using ray-tune for hparam optimisation
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


# def start_without_tuning(root_path='./', model_name = 'ResNet', save_name=None, prv_ckpt_path=None):
#     # print('#### inside the start_without_tuning() method - Start')
#     if save_name is None:
#         save_name = model_name

#     config = {}
#     if(model_name == 'ResNet'):
#         config["num_classes"] = 2  # hparam for the model architecture
#         config["resNet_arch_type"] = 18  # 9, 18, 22, 34
#         config["act_fn_name"] = 'relu'  # tanh, relu, leakyrelu, gelu
#         config["block_name"] = 'ResNetBlock'  # ResNetBlock, PreActResNetBlock
#         config["optimizer_name"] = 'SGD'  # Adam, SGD (FOR RESNET BOTH NEEDS TO BE CHECKED)
#         config["lr"] = 0.1  # For SGD: 0.1; Adam: 1e-3
#         config["momentum"] = 0.9  # hparam for the SGD optimizer
#         config["weight_decay"] = 1e-4  # For SGD: 1e-4; Adam: 1e-4
#         config["batch_size"] =  24
#         config["img_resoln"] = 800  # img_resoln can be 256 or 400 or 800
#         config["dbl_combi_flg"] = False  # flag to indicate whether (prot1, prot2) pair and (prot2, prot1) pair both will be considered in the training data (as they will give rise to 2 different figures)
#         config["spec_type"] = 'human'
#         # 'weighted_sampling' flag is used to indicate whether WeightedRandomSampler would be used 
#         # for the batch selection in the DataLoader. It is useful for the trainiing of the dataset with the skewed class distribution.
#         config["weighted_sampling"] = True
#         config["num_epochs"] = 10
#         config["num_workers"] = os.cpu_count() - 5  # for the DataModule creation

#     result_dir = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/train')
#     try:
#         # check if the result_dir already exists and if not, then create it
#         if not os.path.exists(result_dir):
#             print("The directory: " + result_dir + " does not exist.. Creating it...")
#             os.makedirs(result_dir)
#     except OSError as ex:
#         errorMessage = "Creation of the directory " + result_dir + " failed. Exception is: " + str(ex)
#         raise Exception(errorMessage)
#     else:
#         print("Successfully created the directory %s " % result_dir)

#     train_model(root_path=root_path, result_dir=result_dir, model_name=model_name, save_name=save_name, config=config, prv_ckpt_path=prv_ckpt_path)

#     # perform the final training
#     # print('\n# ############################# performing the final training #############################\n')
#     # img_p2ip_cnn_clf_train_final.start(root_path)

#     # perform the testing 
#     # print('\n# ############################# performing the testing #############################\n')
#     # img_p2ip_cnn_clf_test.start(root_path = root_path)
#     # print('#### inside the start_without_tuning() method - End')
#     print('\n ##################### END OF THE ENTIRE PROCESS ######################')


def ray_hparam_tuning(config = None, root_path='./', result_dir='./', model_name='ResNet', cpus_per_trial=15, gpus_per_trial=1, num_samples=10, scheduler_name='ASHA'):
    print('#### inside the ray_hparam_tuning() method - Start')

    # start_without_tuning(root_path=root_path, model_name='ResNet', save_name='ResNet_temp', prv_ckpt_path=None)

    # ############################# KEY CONCEPT - Start: ############################ #
    # One sample corresponds to one trial. One trial is one instance of a Trainable containing a apecific combination of hyper-params.
    # That specific hyper-param combination is drawn from the hyper-param search space as a sample. So one sample implies one trial.
    # Now one trial can have many epochs and each epoch will have specific number of iterations depending upon the batch size.
    # So effectively each trial could have many iterations. In Ray-Tune, 'iteration' correponds to epoch and 'global-iteration' correponds
    # to the iteration within an epoch.
    # see https://docs.ray.io/en/latest/tune/api_docs/execution.html for details about experiment, sample <=> trial, iteration with example.
    # ############################# KEY CONCEPT - End: ############################ #
    scheduler = None
    if(scheduler_name == 'ASHA'):
        # define an Asynchronous Hyperband scheduler. This scheduler decides whether to stop a trial even before finishing all the iterations.
        # It checks at each iteration which trials are likely to perform badly and stops these trials.
        # This way we don’t waste any resources on bad hyperparameter configurations.
        scheduler = ASHAScheduler(
            max_t=config["num_epochs"] ,
            grace_period=1,
            reduction_factor=2)
    elif(scheduler_name == 'PBT'):
        # Another popular method for hyperparameter tuning, called Population Based Training, perturbs hyperparameters during the training run. 
        # PBT trains a group of models (or agents) in parallel. Periodically, poorly performing models clone the state of the
        # top performers, and a random mutation is applied to their hyperparameters in the hopes of outperforming the current top models.
        scheduler = PopulationBasedTraining(
            perturbation_interval=4,
            hyperparam_mutations={
                "lr": ray.tune.loguniform(1e-4, 1e-1)
                ,"batch_size": [32, 64, 128]
            })
    
    # instantiate a CLIReporter to specify which metrics we would like to see in our output tables in the command line.
    # This is optional, but can be used to make sure our output tables only include information we would like to see.
    reporter = CLIReporter(
        parameter_columns=["resNet_arch_type", "block_name", "optimizer_name", "lr", "batch_size", "dbl_combi_flg"],
        metric_columns=["val_loss", "val_aupr", "training_iteration"],
        max_report_frequency=30)

    # The root_path, artifact_name, num_epochs etc. we pass to the training function are constants.
    # To avoid including them as non-configurable parameters in the config specification, we can use
    # ray.tune.with_parameters to wrap around the training function.
    train_fn_with_parameters = ray.tune.with_parameters(train_model
                                                    , root_path=root_path
                                                    , result_dir=result_dir
                                                    , model_name=model_name
                                                    )
    # specify how many resources Tune should request for each trial. This also includes GPUs.
    # PyTorch Lightning takes care of moving the tgraining to the GPUs. We already made sure that our code is compatible
    # with that, so there’s nothing more to do here other than to specify the number of GPUs we would like to use.
    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}

    # Configuration object for syncing.
    # Configure how checkpoints are sync'd to the scheduler/sampler
    sync_config=ray.train.SyncConfig()

    # rayTune specific configurable parameters for defining the checkpointing strategy.
    # tune_checkpoint_config = CheckpointConfig(
    #                         num_to_keep=1  # Number of checkpoints to keep
    #                         , checkpoint_score_attribute='val_loss'  # Specifies by which attribute to rank the best checkpoint.
    #                         , checkpoint_score_order='min'
    #                         # , checkpoint_frequency=1  # Number of iterations between checkpoints. 
    #                         )

    analysis = ray.tune.run(
        run_or_experiment=train_fn_with_parameters,  # trainable
        storage_path=result_dir,
        resources_per_trial=resources_per_trial,
        metric="val_loss", mode="min",
        # metric="val_accuracy", mode="max",
        config=config,
        num_samples=num_samples,  # Number of times to sample from the hyperparameter space. Defaults to 1.
        # num_samples => number of trials where each trial is one instance of a Trainable. 
        # If this is -1, (virtually) infinite samples are generated until a stopping condition is met.
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_" + str(scheduler_name) + "_resnet",
        resume ="AUTO",  #  "AUTO" will attempt to resume from a checkpoint and otherwise start a new experiment.
                         #  "ERRORED_ONLY" resets and reruns errored trials upon resume - previous trial artifacts will be left untouched.
        reuse_actors=False,  # Whether to reuse actors between different trials when possible. 
        log_to_file=True,  # If true, outputs are written to trialdir/stdout and trialdir/stderr, respectively.
        # checkpoint_config=tune_checkpoint_config,
        keep_checkpoints_num=1,  # Number of checkpoints to keep
        # checkpoint_score_attr='min-loss',  # Specifies by which attribute to rank the best checkpoint.
        checkpoint_score_attr='min-val_loss',  # Specifies by which attribute to rank the best checkpoint.
        sync_config = sync_config,
        max_failures = 1,  # Try to recover a trial at least this many times.
        raise_on_failed_trial = False,  # Raise TuneError if there exists failed trial (of ERROR state) when the experiments complete
        verbose = 3)

    # Analysis result
    print("\n# ##### Analysis result -Start ##### #\n")
    best_trial = analysis.best_trial  # Get best trial
    print("## best trial: ", best_trial)

    best_config = analysis.best_config  # Get best trial's hyperparameters
    print("\n## best trial's hyperparameters: ", best_config)

    experiment_path = analysis.experiment_path  # Get best trial's experiment_path
    print("\n## best trial's experiment_path: ", experiment_path)

    best_checkpoint = analysis.best_checkpoint  # Get best trial's best checkpoint
    print("\n## best trial's best checkpoint: ", best_checkpoint)

    best_result = analysis.best_result  # Get best trial's last results
    print("\n## best trial's last results: ", best_result)

    best_result_df = analysis.best_result_df  # Get best result as pandas dataframe
    best_result_df['num_epochs'] = [config["num_epochs"]]
    # save best_result_df
    best_result_df_name_with_path = os.path.join(result_dir, 'best_result_df.csv')
    best_result_df.to_csv(best_result_df_name_with_path, index=False)

    # Save the best result as a pkl file to be used later
    print("\n Saving the best result as a pkl file to be used later")
    # adding a few more parameters to be used later
    best_result['num_epochs'] = config["num_epochs"]
    best_result['best_config'] = best_config
    best_result['best_checkpoint_path'] = str(analysis.best_checkpoint)
    best_result_artifact_name_with_path = os.path.join(result_dir, 'best_result_dict.pkl')
    joblib.dump(value=best_result,
                filename=best_result_artifact_name_with_path,
                compress=3)
    print("The best_result is saved as: " + best_result_artifact_name_with_path)

    print("\n# ##### Analysis result -End ##### #\n")
    print('#### inside the ray_hparam_tuning() method - End')


def start(root_path='./', model_name = 'ResNet'):
    # print('#### inside the start() method - Start')
    # starting Ray on a single machine
    # ray.init(_temp_dir=f"/home/pralaycs/Shubh/raylog", num_cpus=10, num_gpus=2, runtime_env={"py_modules": [os.path.join(path_root,'proc')]})
    ray.init(runtime_env={"py_modules": [os.path.join(path_root,'proc')]})

    scheduler_name = 'ASHA'  # ASHA or PBT
    config = {}
    if(model_name == 'ResNet'):
        config["num_classes"] = 2  # hparam for the model architecture
        config["resNet_arch_type"] = ray.tune.grid_search([9, 18])  # 9, 18, 22, 34
        config["act_fn_name"] = 'relu'  # tanh, relu, leakyrelu, gelu
        config["block_name"] = ray.tune.grid_search(['ResNetBlock'])  # ResNetBlock, PreActResNetBlock
        config["optimizer_name"] = ray.tune.grid_search(['SGD', 'Adam'])  # Adam, SGD (FOR RESNET BOTH NEEDS TO BE CHECKED)
        # config["lr"] = 0.1  # For SGD: 0.1; Adam: 1e-2
        config["momentum"] = 0.9  # hparam for the SGD optimizer
        config["weight_decay"] = 1e-4  # For SGD: 1e-4; Adam: 1e-4
        config["batch_size"] =  ray.tune.grid_search([24])  # 16, 24
        config["img_resoln"] = 400  # img_resoln can be 400
        config["dbl_combi_flg"] = ray.tune.grid_search([False, True])  # flag to indicate whether (prot1, prot2) pair and (prot2, prot1) pair both will be considered in the training data (as they will give rise to 2 different figures)
        config["spec_type"] = 'human'
        # 'weighted_sampling' flag is used to indicate whether WeightedRandomSampler would be used 
        # for the batch selection in the DataLoader. It is useful for the trainiing of the dataset with the skewed class distribution.
        config["weighted_sampling"] = True
        config["num_epochs"] = 7
        # config["num_workers"] = os.cpu_count() - 5  # for the DataModule creation
        config["num_workers"] = 4  # for the DataModule creation

    # depending upon the sheduler, set 'lr' value in the config
    if(scheduler_name == 'ASHA'):
        # config["lr"] = ray.tune.loguniform(1e-4, 1e-1)  # for ASHA Scheduler
        # for below, refer: https://docs.ray.io/en/latest/tune/tutorials/tune-search-spaces.html#tune-custom-search
        config["lr"] = ray.tune.sample_from(lambda config: 0.1 if(config['optimizer_name']=='SGD') else 0.01),
    elif(scheduler_name == 'PBT'):
        config["lr"] = 1e-2  # for PBT Scheduler
    cpus_per_trial = 4  # cpus_per_trial=1 implies the no. of concurrent trials at a time = the no of CPU (cores) in the machine
    gpus_per_trial = 1 # if you set gpu=1, then only one trial will run at a time but for gpu=0, the concurrent trials will depend on the 'cpus_per_trial'
    num_samples = 1  # 3  # ################### CHANGE AS PER THE REQUIRED RANGE FOR HPARAM TUNING
    # num_epochs = 15  # 25
    result_dir = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/train/resnet_tune_results')
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

    ray_hparam_tuning(config=config, root_path=root_path, result_dir=result_dir, model_name=model_name
                      , cpus_per_trial=cpus_per_trial, gpus_per_trial=gpus_per_trial
                      , num_samples=num_samples, scheduler_name=scheduler_name)

    # perform the final training
    # print('\n# ############################# performing the final training #############################\n')
    # img_p2ip_cnn_clf_train_final.start(root_path)

    # perform the testing 
    # print('\n# ############################# performing the testing #############################\n')
    # img_p2ip_cnn_clf_test.start(root_path = root_path)

    # print('#### inside the start() method - End')
    print('\n ##################### END OF THE ENTIRE PROCESS ######################')


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    # # prv_ckpt_path would be None for the fresh training
    # prv_ckpt_path = None
    # prv_ckpt_path = os.path.join(root_path, 'dataset/proc_data_tl_feat_to_img/img_p2ip_cnn/train/ResNet')
    # start(root_path=root_path, model_name='ResNet', save_name='ResNet_expt_r800n18ND', prv_ckpt_path=prv_ckpt_path)

    start(root_path=root_path, model_name='ResNet')
