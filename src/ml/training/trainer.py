import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
import h5py
from ml.training.loops import train_loop, test_loop
from ml.utils.utils import seed_everything, print_logs, load_model, get_conf_matrix, plot_conf_matrix, get_device
from ml.losses.focal_loss import FocalLoss, make_weights_for_balanced_classes
from ml.data.preprocess_GS import process_image
from ml.model.CNN import CNN
from ml.data.dataset_GS import WaveFormDatasetFast, get_metadata

class Trainer():
    def __init__(self, DATASET_PATH, METADATA_PATH, file_dir, model_class=CNN, loss="baseline", model_params={}, 
                 res=None, epochs=50, batch_size=64, weight_decay= 2*1e-4 ,lr = 0.001, label_smoothing=0, 
                 weighted_loss=False, dataset=WaveFormDatasetFast, process_train=process_image, process_test=process_image,
                 scheduler_active=False, device=get_device(), init_model=False, use_amp = True, patience=2,
                 save_to_ram=True):
        metadata = get_metadata(DATASET_PATH, METADATA_PATH)
        self.names_classes = metadata["names_classes"]
        self.num_classes = len(self.names_classes)
        self.class_counts = metadata["class_counts"]
        
        if res:
            self.training_data = dataset(DATASET_PATH, METADATA_PATH, "train", res=res, transform=process_train, device=device, save_to_ram=save_to_ram)
            self.val_data = dataset(DATASET_PATH, METADATA_PATH, "validation", res=res, transform=process_test, device=device, save_to_ram=save_to_ram)
            self.test_data = dataset(DATASET_PATH, METADATA_PATH, "test", res=res, transform=process_test, device=device, save_to_ram=save_to_ram)
        else:
            self.training_data = dataset(DATASET_PATH, METADATA_PATH, "train", transform=process_train, device=device, save_to_ram=save_to_ram)
            self.val_data = dataset(DATASET_PATH, METADATA_PATH, "validation", transform=process_test, device=device, save_to_ram=save_to_ram)
            self.test_data = dataset(DATASET_PATH, METADATA_PATH, "test", transform=process_test, device=device, save_to_ram=save_to_ram)
        
        self.file_dir = file_dir
        self.model_class = model_class
        self.init_model = init_model
        self.model = None
        self.loss = loss
        self.model_params = model_params
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.lr = lr
        self.label_smoothing = label_smoothing
        self.weighted_loss = weighted_loss
        self.scheduler_active = scheduler_active
        self.device = device
        self.dataset = dataset
        self.process_train = process_train
        self.process_test = process_test
        self.use_amp = use_amp
        self.patience = patience
        self.save_to_ram = save_to_ram
    
    def new_parameters(self, **args):
        # Iterate over the parameters and update the ones that are not None
        for key, value in args.items():
            if value is not None:
                setattr(self, key, value)

    def train_test(self, keep_training=False):
        seed_everything(0)

        if not os.path.exists(self.file_dir):
            os.makedirs(self.file_dir)

        weights = make_weights_for_balanced_classes(self.training_data, self.num_classes)                                                                
        weights = torch.DoubleTensor(weights)                                       
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))  

        # Define dataloaders
        train_dataloader = DataLoader(self.training_data, batch_size=self.batch_size, sampler=train_sampler)
        val_dataloader = DataLoader(self.val_data, batch_size=self.batch_size)
        test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size)

        # Instantiate model
        if not keep_training:
            self.model = self.model_class(**self.model_params)
            self.model = nn.DataParallel(self.model)
            self.model.to(self.device)

        # Initialize model
        if self.init_model:
            self.model.module.init_weights(next(iter(train_dataloader))[0].to(self.device))

        # Define loss and optimizer
        # Define loss and optimizer
        if self.weighted_loss:
            class_proportions = (1-0.99)/(1 - 0.99 ** torch.tensor(self.class_counts))
            class_proportions = (class_proportions/sum(class_proportions)).to(self.device)
        else:
            class_proportions=None
        if self.loss=="baseline":
            loss_fn = nn.CrossEntropyLoss(weight=class_proportions, label_smoothing=self.label_smoothing)
        elif self.loss=="focal":
            loss_fn = FocalLoss(weight=class_proportions)
        #optimizer = torch.optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.scheduler_active:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=True, patience=self.patience, min_lr=1e-8)
        else:
            scheduler=None

        params = {"batch_size":[self.batch_size], "lr":[self.lr], "weight_decay":[self.label_smoothing], "weight_decay":[self.label_smoothing],"weighted_loss":self.weighted_loss,
                         "optimizer":type(optimizer).__name__, "loss":self.loss}
        params.update(self.model_params)

        df = pd.DataFrame.from_dict(params)
        df.to_csv(self.file_dir / "params.csv")

        # Train model
        epochs = self.epochs

        train_log = []
        test_log = []
        best_metric = 0
        test_acc_log = []
        test_f1_log = []

        print("--- Begin training ---")
        for epoch in range(epochs):
            train_loss = train_loop(train_dataloader, self.model, loss_fn, optimizer, epoch=epoch, scheduler=scheduler, max_epochs=epochs, device=self.device, use_amp=self.use_amp)
            test_loss, test_acc, test_f1 = test_loop(val_dataloader, self.model, loss_fn, split="val", device=self.device)
            train_log.append(train_loss)
            test_log.append(test_loss)
            test_acc_log.append(test_acc)
            test_f1_log.append(test_f1)
            if test_f1 > best_metric:
                best_metric = test_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': test_loss,
                    'accuracy': test_acc,
                    'f1-score': test_f1
                    }, self.file_dir / 'model.pt')
        print("Done!")
        df = pd.DataFrame({"train_log":train_log, "test_log":test_log, "test_acc_log":test_acc_log, "test_f1_log":test_f1_log})
        df.to_csv(self.file_dir/"logs.csv")
        print_logs(df, epochs=epochs, file_dir=self.file_dir)

        self.model, best_acc, best_f1 = load_model(self.model_class, self.file_dir / "model.pt", test_dataloader, params=self.model_params, device=self.device, is_parallel=True, init=self.init_model)
        print(f"Best val acc: {(100*best_acc):>0.2f}%, Best val f1: {(100*best_f1):>0.2f}%")

        loss, accuracy, f1 = test_loop(test_dataloader, self.model, loss_fn, split="test", device=self.device)
        df_test = pd.DataFrame({"accuracy":[accuracy], "f1":[f1]})
        df_test.to_csv(self.file_dir/"metrics.csv")

        confusion_matrix = get_conf_matrix(self.model, test_dataloader, len(self.names_classes), device=self.device)
        plot_conf_matrix(confusion_matrix, self.names_classes, self.file_dir)  
