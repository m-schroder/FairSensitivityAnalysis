import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchmetrics import F1Score

from modules import functions
import pickle
import random


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

class FairClfDataset(Dataset):

    def __init__(self, df, label_col):
        self.label_col = label_col
        self.df = df

        labels = df.iloc[:, df.columns == label_col].values
        self.labels = torch.tensor(labels).float()
        covariates = df.iloc[:, df.columns != label_col].values
        self.covariates = torch.tensor(covariates).float()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return (self.covariates[idx], self.labels[idx])



class FairClfDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir,
                 label_col,
                 batch_size=64,
                 num_workers=0, 
                 mode = None):
        super().__init__()
        
        # dataset params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.label_col = label_col
        self.mode = mode

    def setup(self, stage: str):

        with open(self.data_dir, "rb") as input:
            full_data = pickle.load(input)

        if self.mode == "density_M":
            full_data = full_data.drop(["Y", "USE", "UIE", "UDE"],axis=1)
        elif self.mode == "density_A":
            full_data = full_data.drop(["Y", "M", "USE", "UIE", "UDE"],axis=1)
        elif self.mode == "prediction":
            full_data = full_data.drop(["USE", "UIE", "UDE"],axis=1)            
        else:
            raise KeyError('Mode not specified. Please set mode to "prediction" or "density".') 

        self.train_data = FairClfDataset(full_data.iloc[:int(0.6*full_data.shape[0]),:], self.label_col)
        self.val_data = FairClfDataset(full_data.iloc[int(0.6*full_data.shape[0]):int(0.8*full_data.shape[0]),:], self.label_col)
        self.test_data = FairClfDataset(full_data.iloc[int(0.8*full_data.shape[0]):,:], self.label_col)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers = self.num_workers, shuffle = True, worker_init_fn=seed_worker, generator=g)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers = self.num_workers, shuffle = False, worker_init_fn=seed_worker, generator=g)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers = self.num_workers, shuffle = False, worker_init_fn=seed_worker, generator=g)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers = self.num_workers, shuffle = False, worker_init_fn=seed_worker, generator=g)



class Density_estimator(pl.LightningModule):                                      
    def __init__(self, 
                 n_features, 
                 n_classes,
                 hidden_dim,
                 learning_rate,
                 criterion,
                 class_weights,
                 dropout):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.class_weights = torch.tensor(class_weights).double()
        self.criterion = criterion()
        self.learning_rate = learning_rate
        if n_classes == 1:
            self.task = "binary"
            self.accuracy = F1Score(task=self.task)
        else:
            self.task = "multiclass"
            self.accuracy = F1Score(task=self.task, num_classes=n_classes)
        self.save_hyperparameters()

        self.net = nn.Sequential(
            nn.BatchNorm1d(self.n_features),
            nn.Linear(self.n_features, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, n_classes)
            ) 
        self.softmax = nn.Softmax(dim = 1)  
        self.sigmoid = nn.Sigmoid()    
        
    def forward(self, x):
        out = self.net(x)
        if self.task == "multiclass":
            return out, self.softmax(out)
        else:
            return out, self.sigmoid(out)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_logits, y_probs = self(x)
        if self.task == "multiclass":
            y_hat = torch.argmax(y_probs, dim = 1, keepdim=True).double()
            loss = self.criterion(y_probs, y.squeeze().long())
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            y_hat = torch.argmax(y_logits, dim = 1, keepdim=True).long()
            class_weights = torch.gather(self.class_weights.repeat(y_hat.size()), dim = 1, index=y_hat)
            criterion = nn.BCEWithLogitsLoss(weight=class_weights)
            loss = criterion(y_logits, y)
            step_accuracy = self.accuracy(y_probs, y)
        self.log("train_loss", loss, logger = True, prog_bar=False)
        self.log("train_accuracy", step_accuracy, logger = True, prog_bar=False)
        return {"loss": loss, "accuracy": step_accuracy}

    def training_epoch_end(self, outputs):    
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Train loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Train accuracy per epoch", avg_acc, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_logits, y_probs = self(x)
        if self.task == "multiclass":
            y_hat = torch.argmax(y_probs, dim = 1, keepdim=True).double()
            loss = self.criterion(y_probs, y.squeeze().long())
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            y_hat = torch.argmax(y_logits, dim = 1, keepdim=True).long()
            class_weights = torch.gather(self.class_weights.repeat(y_hat.size()), dim = 1, index=y_hat)
            criterion = nn.BCEWithLogitsLoss(weight=class_weights)
            loss = criterion(y_logits, y)
            step_accuracy = self.accuracy(y_probs, y)
        self.log("val_loss", loss, logger = True, prog_bar=False)
        self.log("val_accuracy", step_accuracy, logger = True, prog_bar=False)
        return {"loss": loss, "accuracy": step_accuracy}

    def validation_epoch_end(self, outputs):        
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Validation loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Validation accuracy per epoch", avg_acc, self.current_epoch)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_logits, y_probs = self(x)
        if self.task == "multiclass":
            y_hat = torch.argmax(y_probs, dim = 1, keepdim=True).double()
            loss = self.criterion(y_probs, y.squeeze().long())
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            y_hat = torch.argmax(y_logits, dim = 1, keepdim=True).long()
            class_weights = torch.gather(self.class_weights.repeat(y_hat.size()), dim = 1, index=y_hat)
            criterion = nn.BCEWithLogitsLoss(weight=class_weights)
            loss = criterion(y_logits, y)
            step_accuracy = self.accuracy(y_probs, y)
        self.log("test_loss", loss, logger = True, prog_bar=False)
        self.log("test_accuracy", step_accuracy, logger = True, prog_bar=False)
        return {"loss": loss, "accuracy": step_accuracy}
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_probs = self(x)
        if self.task == "multiclass":
            y_hat = torch.argmax(y_probs, dim = 1, keepdim=True).double()
            loss = self.criterion(y_probs, y.squeeze().long())
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            loss = self.criterion(y_probs, y)
            step_accuracy = self.accuracy(y_probs, y)
        return {"loss": loss, "accuracy": step_accuracy}
    






# -----------------------------------------------------------------Classifier------------------------------------------------------------------------------------


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)

class StandardClf(pl.LightningModule):                                      
    def __init__(self, 
                 n_features, 
                 n_classes,
                 hidden_dim,
                 learning_rate,
                 criterion,
                 class_weights,
                 dropout):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.class_weights = torch.tensor(class_weights).float()
        self.criterion = criterion()
        self.learning_rate = learning_rate
        if n_classes == 1:
            self.task = "binary"
            self.accuracy = F1Score(task=self.task)
        else:
            self.task = "multiclass"
            self.accuracy = F1Score(task=self.task, num_classes=n_classes)   
        self.save_hyperparameters()

        self.net = nn.Sequential(
            nn.BatchNorm1d(self.n_features),
            nn.Linear(self.n_features, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, n_classes)
            ) 
        self.softmax = nn.Softmax(dim = 1)  
        self.sigmoid = nn.Sigmoid()   
        self.net.apply(init_weights) 
        
    def forward(self, x):
        out = self.net(x)
        if self.task == "multiclass":
            return self.softmax(out)
        else:
            return out, self.sigmoid(out)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.task == "multiclass":
            y_probs = self(x)
            y_hat = torch.argmax(y_probs, dim = 1, keepdim=True).double()
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = criterion(y_probs.to(torch.float32), y.squeeze().long())
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            y_logits, y_probs = self(x)
            y_hat = torch.argmax(y_logits, dim = 1, keepdim=True).long()
            class_weights = torch.gather(self.class_weights.repeat(y_logits.size()), dim = 1, index=y_hat)
            criterion = nn.BCEWithLogitsLoss(weight=class_weights)
            loss = criterion(y_logits, y)
            step_accuracy = self.accuracy(y_probs, y)
        self.log("train_loss", loss, logger = True, prog_bar=False)
        self.log("train_accuracy", step_accuracy, logger = True, prog_bar=False)
        return {"loss": loss, "accuracy": step_accuracy}

    def training_epoch_end(self, outputs):    
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Train loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Train accuracy per epoch", avg_acc, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.task == "multiclass":
            y_probs = self(x)
            y_hat = torch.argmax(y_probs, dim = 1, keepdim=True).double()
            criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = criterion(y_probs.to(torch.float32), y.squeeze().long())
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            y_logits, y_probs = self(x)
            y_hat = torch.argmax(y_logits, dim = 1, keepdim=True).long()
            class_weights = torch.gather(self.class_weights.repeat(y_logits.size()), dim = 1, index=y_hat)
            criterion = nn.BCEWithLogitsLoss(weight=class_weights)
            loss = criterion(y_logits, y)
            step_accuracy = self.accuracy(y_probs, y)
        self.log("val_loss", loss, logger = True, prog_bar=False)
        self.log("val_accuracy", step_accuracy, logger = True, prog_bar=False)
        return {"loss": loss, "accuracy": step_accuracy}

    def validation_epoch_end(self, outputs):        
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Validation loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Validation accuracy per epoch", avg_acc, self.current_epoch)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.task == "multiclass":
            y_probs = self(x)
            y_hat = torch.argmax(y_probs, dim = 1, keepdim=True).double()
            class_weights = torch.gather(self.class_weights.repeat(y_hat.unsqueeze(-1).size()), dim = 1, index=y_hat.unsqueeze(-1))
            criterion = nn.BCELoss(weight=class_weights)
            loss = criterion(y_probs.to(torch.float32), y.squeeze().long())
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            y_logits, y_probs = self(x)
            y_hat = torch.argmax(y_logits, dim = 1, keepdim=True).long()
            class_weights = torch.gather(self.class_weights.repeat(y_logits.size()), dim = 1, index=y_hat)
            self.criterion = nn.BCELoss(weight=class_weights)
            loss = self.criterion(y_logits, y)
            step_accuracy = self.accuracy(y_logits, y)
        self.log("test_loss", loss, logger = True, prog_bar=False)
        self.log("test_accuracy", step_accuracy, logger = True, prog_bar=False)
        return {"loss": loss, "accuracy": step_accuracy}

        

class FairClf_lambda(pl.LightningModule):                                      
    def __init__(self, 
                 n_features, 
                 n_classes,
                 train_data, 
                 batch_size,
                 hidden_dim,
                 learning_rate,
                 sensitive_attributes, #lists all values of A
                 sensitivity_parameter,
                 constraints, # of form (gamma_DE, gamma_IE, gamma_SE)
                 criterion_pred,
                 nested_epochs,
                 checkpointpath_density,
                 checkpointpath_propensity,
                 class_weights,
                 dropout,
                 column_names):
        super().__init__()
        self.n_features = n_features
        self.n_classes = np.int(n_classes)
        self.train_data = train_data
        self.batch_size = batch_size
        self.criterion_pred = criterion_pred()
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.class_weights = torch.tensor(class_weights).float()
        self.sensitive_attributes = sensitive_attributes
        self.sensitivity_parameter = sensitivity_parameter
        self.constraints = torch.tensor(constraints, requires_grad=True)
        self.checkpointpath_density = checkpointpath_density
        self.checkpointpath_propensity = checkpointpath_propensity
        self.column_names = column_names
        self.nested_epochs = nested_epochs
        # replace by chosen penalty parameters
        self.lambdas = torch.tensor([0.1, 0.1, 0.1], requires_grad=True)
        self.task = None
        if self.n_classes == 1:
            self.task = "binary"
            self.accuracy = F1Score(task=self.task)
        else:
            self.task = "multiclass"
            self.accuracy = F1Score(task=self.task, num_classes=n_classes)
        self.save_hyperparameters()

        self.automatic_optimization = False
        torch.autograd.set_detect_anomaly(True)

        self.net = nn.Sequential(
            nn.Linear(self.n_features, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, n_classes)) 
        self.softmax = nn.Softmax(dim = 1)     
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.net(x)
        if self.task == "multiclass":
            return self.softmax(out)
        else:
            return out, self.sigmoid(out)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
    
    def constraint_function(self, task, path, path_prop):

        if self.checkpointpath_density == None:
            density_estimator = None
        else:
            density_estimator = Density_estimator.load_from_checkpoint(path).eval()
        if self.checkpointpath_propensity == None:
            pass
        else:
            propensity_model = Density_estimator.load_from_checkpoint(path_prop).eval()

        ctf_effects = functions.Ctf_effects_binary_torch(network=self.forward, data = self.train_data, task = task, sensitivity_param=self.sensitivity_parameter, density_estimator=density_estimator, propensity_model=propensity_model)

        y_ind = 1.0
        ub_DE_y, lb_DE_y = ctf_effects.DE_binary(self.sensitive_attributes[0], self.sensitive_attributes[1], y_ind)
        ub_IE_y, lb_IE_y = ctf_effects.IE_binary(self.sensitive_attributes[0], self.sensitive_attributes[1], y_ind)
        ub_SE_y, lb_SE_y = ctf_effects.SE_binary(self.sensitive_attributes[0], self.sensitive_attributes[1], y_ind) 
        bounds = torch.tensor([ub_DE_y, lb_DE_y, ub_IE_y, lb_IE_y, ub_SE_y, lb_SE_y], requires_grad=True)

        bound_DE = torch.max(bounds[0:2]).requires_grad_()
        bound_IE = torch.max(bounds[2:4]).requires_grad_()
        bound_SE = torch.max(bounds[4:6]).requires_grad_()

        max_bounds = torch.tensor([bound_DE, bound_IE, bound_SE], requires_grad=True)
        return (self.constraints - max_bounds).requires_grad_(), max_bounds.requires_grad_()

    def training_step(self, batch, batch_idx):    
        x, y = batch
        if self.task == "multiclass":
            y_probs = self(x)
            y_hat = torch.argmax(y_probs, dim = 1, keepdim=True).double()
            criterion_pred = nn.CrossEntropyLoss(weight=self.class_weights)
            loss_pred = criterion_pred(y_probs.to(torch.float32), y.squeeze().long())
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            logits, y_probs = self(x)
            y_hat = torch.round(y_probs).long()
            class_weights = torch.gather(self.class_weights.repeat(y_hat.size()), dim = 1, index=y_hat)
            criterion_pred = nn.BCEWithLogitsLoss(weight=class_weights)
            loss_pred = criterion_pred(logits, y)
            step_accuracy = self.accuracy(y_probs, y)

        constr, effects = self.constraint_function(self.task, self.checkpointpath_density, self.checkpointpath_propensity)
        loss = loss_pred - torch.sum(self.lambdas * constr).requires_grad_() 

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)           
        opt.step() 

        if self.task == "multiclass":
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            step_accuracy = self.accuracy(y_probs, y)

        fairness = effects
        self.log("train_loss", loss, logger = True, prog_bar=True)
        self.log("train_loss_pred", loss_pred, logger = True, prog_bar=True)
        self.log("train_accuracy", step_accuracy, logger = True, prog_bar=True)
        self.log("train_fairness_DE", fairness[0], logger = True)
        self.log("train_fairness_IE", fairness[1], logger = True)
        self.log("train_fairness_SE", fairness[2], logger = True)
        self.log("train_fairness", torch.max(fairness), logger = True)
        return {"loss": loss, "loss_pred": loss_pred, "accuracy": step_accuracy, "train_fairness": torch.max(fairness)}


    def training_epoch_end(self, outputs):   
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        fairness = torch.stack([x["train_fairness"] for x in outputs]).mean(dim=0)
        self.logger.experiment.add_scalar("Train loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Train accuracy per epoch", avg_acc, self.current_epoch)
        self.logger.experiment.add_scalar("Train fairness per epoch", fairness, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.task == "multiclass":
            y_probs = self(x)
            y_hat = torch.argmax(y_probs, dim = 1, keepdim=True).double()
            criterion_pred = nn.CrossEntropyLoss(weight=self.class_weights)
            loss_pred = criterion_pred(y_probs.to(torch.float32), y.squeeze().long())
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            logits, y_probs = self(x)
            y_hat = torch.round(y_probs).long()
            class_weights = torch.gather(self.class_weights.repeat(y_hat.size()), dim = 1, index=y_hat)
            criterion_pred = nn.BCEWithLogitsLoss(weight=class_weights)
            loss_pred = criterion_pred(logits, y)
            step_accuracy = self.accuracy(y_probs, y)
        
        constr, effects = self.constraint_function(self.task, self.checkpointpath_density, self.checkpointpath_propensity)
        loss = loss_pred.requires_grad_() 
        
        fairness = effects
        self.log("val_loss", loss, logger = True)
        self.log("val_loss_pred", loss_pred, logger = True)
        self.log("val_accuracy", step_accuracy, logger = True)
        self.log("val_fairness_DE", fairness[0], logger = True)
        self.log("val_fairness_IE", fairness[1], logger = True)
        self.log("val_fairness_SE", fairness[2], logger = True)
        self.log("val_fairness", torch.max(fairness), logger = True)
        return {"loss": loss, "loss_pred": loss_pred, "accuracy": step_accuracy, "fairness": torch.max(fairness)}

    def validation_epoch_end(self, outputs):   
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        avg_fairness = torch.stack([x["fairness"] for x in outputs]).mean()        
        self.logger.experiment.add_scalar("Validation loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Validation accuracy per epoch", avg_acc, self.current_epoch)
        self.logger.experiment.add_scalar("Validation fairness per epoch", avg_acc, self.current_epoch)
        print("val_fairness" + str(avg_fairness))
        return{"val_loss": avg_loss, "val_fairness": avg_fairness}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.task == "multiclass":
            y_probs = self(x)
            y_hat = torch.argmax(y_probs, dim = 1, keepdim=True).double()
            criterion_pred = nn.CrossEntropyLoss(weight=self.class_weights)
            loss_pred = criterion_pred(y_probs.to(torch.float32), y.squeeze().long())
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            logits, y_probs = self(x)
            y_hat = torch.round(y_probs).long()
            class_weights = torch.gather(self.class_weights.repeat(y_hat.size()), dim = 1, index=y_hat)
            criterion_pred = nn.BCEWithLogitsLoss(weight=class_weights)
            loss_pred = criterion_pred(logits, y)
            step_accuracy = self.accuracy(y_probs, y)
        
        self.constr = self.constraint_function(self.task, self.checkpointpath_density, self.checkpointpath_propensity)
        loss_fairness = torch.sum(self.lambdas * self.constr[0])
        loss = loss_pred - loss_fairness

        fairness = self.constr[1]
        self.log("test_loss", loss, logger = True, prog_bar=True)
        self.log("test_loss_pred", loss_pred, logger = True, prog_bar=True)
        self.log("test_accuracy", step_accuracy, logger = True, prog_bar=True)
        self.log("test_fairness_DE", fairness[0], logger = True)
        self.log("test_fairness_IE", fairness[1], logger = True)
        self.log("test_fairness_SE", fairness[2], logger = True)
        self.log("test_fairness", torch.max(fairness), logger = True)        
        return {"loss": loss, "loss_pred": loss_pred, "accuracy": step_accuracy, "fairness": torch.max(fairness)}



class FairClf(pl.LightningModule):                                      
    def __init__(self, 
                 n_features, 
                 n_classes,
                 train_data, 
                 batch_size,
                 hidden_dim,
                 learning_rate,
                 sensitive_attributes, #lists all values of A
                 sensitivity_parameter,
                 constraints, # of form (gamma_DE, gamma_IE, gamma_SE)
                 criterion_pred,
                 nested_epochs,
                 checkpointpath_density,
                 class_weights,
                 dropout,
                 column_names,
                 checkpointpath_propensity = None,
                 ):
        super().__init__()
        self.n_features = n_features
        self.n_classes = np.int(n_classes)
        self.train_data = train_data
        self.batch_size = batch_size
        self.criterion_pred = criterion_pred()
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.class_weights = torch.tensor(class_weights).float()
        self.sensitive_attributes = sensitive_attributes
        self.sensitivity_parameter = sensitivity_parameter
        self.constraints = torch.tensor(constraints, requires_grad=True)
        self.checkpointpath_density = checkpointpath_density
        self.checkpointpath_propensity = checkpointpath_propensity
        self.column_names = column_names
        self.nested_epochs = nested_epochs
         # replace by chosen lagrangian parameters
        self.lambdas = torch.tensor([0.2, 0.2, 0.2], requires_grad=True)        
        self.lambdas_old = self.lambdas
        self.mu = 0.5
        self.update_mu = 1.5
        self.task = None
        if self.n_classes == 1:
            self.task = "binary"
            self.accuracy = F1Score(task=self.task)
        else:
            self.task = "multiclass"
            self.accuracy = F1Score(task=self.task, num_classes=n_classes)
        self.save_hyperparameters()

        self.automatic_optimization = False
        torch.autograd.set_detect_anomaly(True)

        self.net = nn.Sequential(
            nn.Linear(self.n_features, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, n_classes)) 
        self.softmax = nn.Softmax(dim = 1)     
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.net(x)
        if self.task == "multiclass":
            return self.softmax(out)
        else:
            return out, self.sigmoid(out)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
    
    def constraint_function(self, task, path, path_prop):
        
        density_estimator = Density_estimator.load_from_checkpoint(path).eval()
        if path_prop == None:
            propensity_model = None
        else:
            propensity_model = Density_estimator.load_from_checkpoint(path_prop).eval()

        ctf_effects = functions.Ctf_effects_binary_torch(network=self, data = self.train_data, task = self.task, sensitivity_param=self.sensitivity_parameter, density_estimator=density_estimator, propensity_model=propensity_model)

        y_ind = 1.0
        ub_DE_y, lb_DE_y = ctf_effects.DE_binary(self.sensitive_attributes[0], self.sensitive_attributes[1], y_ind)
        ub_IE_y, lb_IE_y = ctf_effects.IE_binary(self.sensitive_attributes[0], self.sensitive_attributes[1], y_ind)
        ub_SE_y, lb_SE_y = ctf_effects.SE_binary(self.sensitive_attributes[0], self.sensitive_attributes[1], y_ind)  
        bounds = torch.tensor([ub_DE_y, lb_DE_y, ub_IE_y, lb_IE_y, ub_SE_y, lb_SE_y], requires_grad=True)
        bound_DE = torch.max(torch.abs(bounds[0:2]).requires_grad_()).requires_grad_()
        bound_IE = torch.max(torch.abs(bounds[2:4]).requires_grad_()).requires_grad_()
        bound_SE = torch.max(torch.abs(bounds[4:6]).requires_grad_()).requires_grad_()

        max_bounds = torch.tensor([bound_DE, bound_IE, bound_SE], requires_grad=True)
        return (self.constraints - max_bounds).requires_grad_(), max_bounds.requires_grad_()

    def training_step(self, batch, batch_idx):    
        x, y = batch
        if self.task == "multiclass":
            y_probs = self(x)
            y_hat = torch.argmax(y_probs, dim = 1, keepdim=True).double()
            criterion_pred = nn.CrossEntropyLoss(weight=self.class_weights)
            loss_pred = criterion_pred(y_probs.to(torch.float32), y.squeeze().long())
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            logits, y_probs = self(x)
            y_hat = torch.round(y_probs).long()
            class_weights = torch.gather(self.class_weights.repeat(y_hat.size()), dim = 1, index=y_hat)
            criterion_pred = nn.BCEWithLogitsLoss(weight=class_weights)
            loss_pred = criterion_pred(logits, y)
            step_accuracy = self.accuracy(y_probs, y)

        self.constr = self.constraint_function(self.task, self.checkpointpath_density, self.checkpointpath_propensity)
        loss = loss_pred - torch.sum(self.lambdas * self.constr[0]).requires_grad_() + self.mu/2 * torch.sum(self.lambdas -self.lambdas_old).requires_grad_()**2
        
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)           
        opt.step() 

        if self.task == "multiclass":
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            step_accuracy = self.accuracy(y_probs, y)

        fairness = self.constr[1]
        self.log("train_loss", loss, logger = True, prog_bar=True)
        self.log("train_loss_pred", loss_pred, logger = True, prog_bar=True)
        self.log("train_accuracy", step_accuracy, logger = True, prog_bar=True)
        self.log("train_fairness_DE", fairness[0], logger = True)
        self.log("train_fairness_IE", fairness[1], logger = True)
        self.log("train_fairness_SE", fairness[2], logger = True)
        self.log("train_fairness", torch.max(fairness), logger = True)
        return {"loss": loss, "loss_pred": loss_pred, "accuracy": step_accuracy, "train_fairness": torch.max(fairness)}


    def training_epoch_end(self, outputs):   

        if self.current_epoch % self.nested_epochs == 0 and self.current_epoch != 0:    
            self.lambdas_old = self.lambdas
            self.lambdas.data = np.maximum(self.lambdas.data - self.mu * self.constr[0].detach().numpy(), np.zeros(len(self.lambdas.data)))   
            self.mu = self.update_mu*self.mu

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        fairness = torch.stack([x["train_fairness"] for x in outputs]).mean(dim=0)
        self.logger.experiment.add_scalar("Train loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Train accuracy per epoch", avg_acc, self.current_epoch)
        self.logger.experiment.add_scalar("Train fairness per epoch", fairness, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.task == "multiclass":
            y_probs = self(x)
            y_hat = torch.argmax(y_probs, dim = 1, keepdim=True).double()
            criterion_pred = nn.CrossEntropyLoss(weight=self.class_weights)
            loss_pred = criterion_pred(y_probs.to(torch.float32), y.squeeze().long())
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            logits, y_probs = self(x)
            y_hat = torch.round(y_probs).long()
            class_weights = torch.gather(self.class_weights.repeat(y_hat.size()), dim = 1, index=y_hat)
            criterion_pred = nn.BCEWithLogitsLoss(weight=class_weights)
            loss_pred = criterion_pred(logits, y)
            step_accuracy = self.accuracy(y_probs, y)
        
        self.constr = self.constraint_function(self.task, self.checkpointpath_density, self.checkpointpath_propensity)
        loss = loss_pred - torch.sum(self.lambdas * self.constr[0]).requires_grad_() + self.mu/2 * torch.sum(self.lambdas -self.lambdas_old).requires_grad_()**2
        
        fairness = self.constr[1]
        self.log("val_loss", loss, logger = True)
        self.log("val_loss_pred", loss_pred, logger = True)
        self.log("val_accuracy", step_accuracy, logger = True)
        self.log("val_fairness_DE", fairness[0], logger = True)
        self.log("val_fairness_IE", fairness[1], logger = True)
        self.log("val_fairness_SE", fairness[2], logger = True)
        self.log("val_fairness", torch.max(fairness), logger = True)
        return {"loss": loss, "loss_pred": loss_pred, "accuracy": step_accuracy, "fairness": torch.max(fairness)}

    def validation_epoch_end(self, outputs):   
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        avg_fairness = torch.stack([x["fairness"] for x in outputs]).mean()        
        self.logger.experiment.add_scalar("Validation loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Validation accuracy per epoch", avg_acc, self.current_epoch)
        self.logger.experiment.add_scalar("Validation fairness per epoch", avg_acc, self.current_epoch)
        print("val_fairness" + str(avg_fairness))
        return{"val_loss": avg_loss, "val_fairness": avg_fairness}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.task == "multiclass":
            y_probs = self(x)
            y_hat = torch.argmax(y_probs, dim = 1, keepdim=True).double()
            criterion_pred = nn.CrossEntropyLoss(weight=self.class_weights)
            loss_pred = criterion_pred(y_probs.to(torch.float32), y.squeeze().long())
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            logits, y_probs = self(x)
            y_hat = torch.round(y_probs).long()
            class_weights = torch.gather(self.class_weights.repeat(y_hat.size()), dim = 1, index=y_hat)
            criterion_pred = nn.BCEWithLogitsLoss(weight=class_weights)
            loss_pred = criterion_pred(logits, y)
            step_accuracy = self.accuracy(y_probs, y)
        
        self.constr = self.constraint_function(self.task, self.checkpointpath_density, self.checkpointpath_propensity)
        loss_fairness = torch.sum(self.lambdas * self.constr[0])
        loss = loss_pred - loss_fairness

        fairness = self.constr[1]
        self.log("test_loss", loss, logger = True, prog_bar=True)
        self.log("test_loss_pred", loss_pred, logger = True, prog_bar=True)
        self.log("test_accuracy", step_accuracy, logger = True, prog_bar=True)
        self.log("test_fairness_DE", fairness[0], logger = True)
        self.log("test_fairness_IE", fairness[1], logger = True)
        self.log("test_fairness_SE", fairness[2], logger = True)
        self.log("test_fairness", torch.max(fairness), logger = True)        
        return {"loss": loss, "loss_pred": loss_pred, "accuracy": step_accuracy, "fairness": torch.max(fairness)}




class FairClf_naive(pl.LightningModule):                                      
    def __init__(self, 
                 n_features, 
                 n_classes,
                 train_data, 
                 batch_size,
                 hidden_dim,
                 learning_rate,
                 sensitive_attributes, #lists all values of A
                 constraints, # of form (gamma_DE, gamma_IE, gamma_SE)
                 density_path,
                 criterion_pred,
                 nested_epochs,
                 class_weights,
                 dropout, 
                 column_names):
        super().__init__()
        self.n_features = n_features
        self.n_classes = np.int(n_classes)
        self.train_data = train_data
        self.batch_size = batch_size
        self.criterion_pred = criterion_pred()
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.checkpoints_density = density_path
        self.class_weights = torch.tensor(class_weights).float()
        self.sensitive_attributes = sensitive_attributes
        self.constraints = torch.tensor(constraints, requires_grad=True)
        self.column_names = column_names
        self.nested_epochs = nested_epochs
        self.lambdas = torch.tensor([3.0, 3.0, 3.0], requires_grad=True)
        self.lambdas_old = self.lambdas
        self.mu = 0.5
        self.update_mu = 1.5
        self.task = None
        if self.n_classes == 1:
            self.task = "binary"
            self.accuracy = F1Score(task=self.task)
        else:
            self.task = "multiclass"
            self.accuracy = F1Score(task=self.task, num_classes=n_classes)
        self.save_hyperparameters()

        self.automatic_optimization = False

        self.net = nn.Sequential(
            nn.Linear(self.n_features, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, n_classes)) 
        self.softmax = nn.Softmax(dim = 1)     
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.net(x)
        if self.task == "multiclass":
            return self.softmax(out)
        else:
            return out, self.sigmoid(out)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
    
    def constraint_function(self, x, y, task):
        if self.checkpoints_density == None:
            density_estimator = None
        else:
            density_estimator = Density_estimator.load_from_checkpoint(self.checkpoints_density).eval()

        ctf_effects = functions.Ctf_effects_binary_torch(network=self.forward, data = self.train_data, task = task, sensitivity_param=None, density_estimator=density_estimator)

        y_ind = 1.0
        new_effects = ctf_effects.unbounded_effects(self.sensitive_attributes[0], self.sensitive_attributes[1], y_ind)
        effects = new_effects.unsqueeze(dim=0)
        abs_effects = torch.abs(effects)
        effects = torch.sum(abs_effects, dim = 0)

        return (self.constraints - effects), effects


    def training_step(self, batch, batch_idx):
        
        x, y = batch

        self.constr = self.constraint_function(x, y, self.task)
        if self.task == "multiclass":
            y_probs = self(x)
            y_hat = torch.argmax(y_probs, dim = 1, keepdim=True).double()
            criterion_pred = nn.CrossEntropyLoss(weight=self.class_weights)
            loss_pred = criterion_pred(y_probs.to(torch.float32), y.squeeze().long())
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            logits, y_probs = self(x)
            y_hat = torch.round(y_probs).long()
            class_weights = torch.gather(self.class_weights.repeat(y_hat.size()), dim = 1, index=y_hat)
            criterion_pred = nn.BCEWithLogitsLoss(weight=class_weights)
            loss_pred = criterion_pred(logits, y)
            step_accuracy = self.accuracy(y_probs, y)

        loss = ((loss_pred - torch.sum(self.lambdas * self.constr[0])) + self.mu/2 * torch.sum(self.lambdas -self.lambdas_old)**2).requires_grad_()

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)            
        opt.step()    

        if self.task == "multiclass":
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            step_accuracy = self.accuracy(y_probs, y)

        fairness =  self.constr[1]
        self.log("train_fairness", torch.max(fairness), logger = True)
        self.log("train_loss", loss, logger = True, prog_bar=True)
        self.log("train_loss_pred", loss_pred, logger = True, prog_bar=True)
        self.log("train_accuracy", step_accuracy, logger = True, prog_bar=True)
        self.log("train_fairness_DE", fairness[0], logger = True)
        self.log("train_fairness_IE", fairness[1], logger = True)
        self.log("train_fairness_SE", fairness[2], logger = True)
        return {"train_fairness": torch.max(fairness), "loss": loss, "loss_pred": loss_pred, "accuracy": step_accuracy}

    def training_epoch_end(self, outputs):    
                
        if self.current_epoch % self.nested_epochs == 0 and self.current_epoch != 0:        
            with torch.no_grad():
                self.lambdas_old = self.lambdas
                self.lambdas.data = np.maximum(self.lambdas.data - self.mu * self.constr[0].detach().numpy(), np.zeros(len(self.lambdas.data)))   
                self.mu = self.update_mu*self.mu

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_loss_pred = torch.stack([x["loss_pred"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        fairness = torch.stack([x["train_fairness"] for x in outputs]).mean(dim=0)
        self.logger.experiment.add_scalar("Train loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Train accuracy per epoch", avg_acc, self.current_epoch)
        self.logger.experiment.add_scalar("Train fairness per epoch", fairness, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.task == "multiclass":
            y_probs = self(x)
            y_hat = torch.argmax(y_probs, dim = 1, keepdim=True).double()
            criterion_pred = nn.CrossEntropyLoss(weight=self.class_weights)
            loss_pred = criterion_pred(y_probs.to(torch.float32), y.squeeze().long())
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            logits, y_probs = self(x)
            y_hat = torch.round(y_probs).long()
            class_weights = torch.gather(self.class_weights.repeat(y_hat.size()), dim = 1, index=y_hat)
            criterion_pred = nn.BCEWithLogitsLoss(weight=class_weights)
            loss_pred = criterion_pred(logits, y)
            step_accuracy = self.accuracy(y_probs, y)

        self.constr = self.constraint_function(x, y, self.task)
        loss = ((loss_pred - torch.sum(self.lambdas * self.constr[0])) + self.mu/2 * torch.sum(self.lambdas -self.lambdas_old)**2).requires_grad_()

        fairness = self.constr[1]
        self.log("val_loss", loss, logger = True)
        self.log("val_loss_pred", loss_pred, logger = True)
        self.log("val_accuracy", step_accuracy, logger = True)
        self.log("val_fairness_DE", fairness[0], logger = True)
        self.log("val_fairness_IE", fairness[1], logger = True)
        self.log("val_fairness_SE", fairness[2], logger = True)
        self.log("val_fairness", torch.max(fairness), logger = True)
        return {"val_loss": loss, "loss_pred": loss_pred, "accuracy": step_accuracy, "val_fairness": torch.max(fairness)}

    def validation_epoch_end(self, outputs):   
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        avg_fairness = torch.stack([x["val_fairness"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Validation loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Validation accuracy per epoch", avg_acc, self.current_epoch)
        self.logger.experiment.add_scalar("Validation fairness per epoch", avg_fairness, self.current_epoch)
        return{"val_loss": avg_loss}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.task == "multiclass":
            y_probs = self(x)
            y_hat = torch.argmax(y_probs, dim = 1, keepdim=True).double()
            criterion_pred = nn.CrossEntropyLoss(weight=self.class_weights)
            loss_pred = criterion_pred(y_probs.to(torch.float32), y.squeeze().long())
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            logits, y_probs = self(x)
            y_hat = torch.round(y_probs).long()
            class_weights = torch.gather(self.class_weights.repeat(y_hat.size()), dim = 1, index=y_hat)
            criterion_pred = nn.BCEWithLogitsLoss(weight=class_weights)
            loss_pred = criterion_pred(logits, y)
            step_accuracy = self.accuracy(y_probs, y)

        self.constr = self.constraint_function(x, y, self.task)
        loss_fairness = torch.sum(torch.tensor(self.lambdas) * self.constr[0])
        loss = loss_pred - loss_fairness

        fairness = self.constr[1]
        self.log("test_loss", loss, logger = True, prog_bar=True)
        self.log("test_loss_pred", loss_pred, logger = True, prog_bar=True)
        self.log("test_accuracy", step_accuracy, logger = True, prog_bar=True)
        self.log("test_fairness_DE", fairness[0], logger = True)
        self.log("test_fairness_IE", fairness[1], logger = True)
        self.log("test_fairness_SE", fairness[2], logger = True)
        return {"loss": loss, "loss_pred": loss_pred, "accuracy": step_accuracy, "fairness": fairness}
    


class FairClf_naive_lambda(pl.LightningModule):                                      
    def __init__(self, 
                 n_features, 
                 n_classes,
                 train_data, 
                 batch_size,
                 hidden_dim,
                 learning_rate,
                 sensitive_attributes, #lists all values of A
                 constraints, # of form (gamma_DE, gamma_IE, gamma_SE)
                 density_path,
                 criterion_pred,
                 nested_epochs,
                 class_weights,
                 dropout, 
                 column_names):
        super().__init__()
        self.n_features = n_features
        self.n_classes = np.int(n_classes)
        self.train_data = train_data
        self.batch_size = batch_size
        self.criterion_pred = criterion_pred()
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.checkpoints_density = density_path
        self.class_weights = torch.tensor(class_weights).float()
        self.sensitive_attributes = sensitive_attributes
        self.constraints = torch.tensor(constraints, requires_grad=True)
        self.column_names = column_names
        self.nested_epochs = nested_epochs
        # replace by chosen penalty parameters
        self.lambdas = torch.tensor([2.0, 2.0, 2.0], requires_grad=True)
        self.task = None
        if self.n_classes == 1:
            self.task = "binary"
            self.accuracy = F1Score(task=self.task)
        else:
            self.task = "multiclass"
            self.accuracy = F1Score(task=self.task, num_classes=n_classes)
        self.save_hyperparameters()

        self.automatic_optimization = False

        self.net = nn.Sequential(
            nn.Linear(self.n_features, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, n_classes)) 
        self.softmax = nn.Softmax(dim = 1)     
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.net(x)
        if self.task == "multiclass":
            return self.softmax(out)
        else:
            return out, self.sigmoid(out)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
    
    def constraint_function(self, x, y, task):
        density_estimator = Density_estimator.load_from_checkpoint(self.checkpoints_density)

        ctf_effects = functions.Ctf_effects_binary_torch(network=self.forward, data = self.train_data, task = task, sensitivity_param=None, density_estimator=density_estimator)

        y_ind = 1.0
        new_effects = ctf_effects.unbounded_effects(self.sensitive_attributes[0], self.sensitive_attributes[1], y_ind)
        effects = new_effects.unsqueeze(dim=0)
        abs_effects = torch.abs(effects)
        effects = torch.sum(abs_effects, dim = 0)

        return (self.constraints - effects), effects


    def training_step(self, batch, batch_idx):
        
        x, y = batch

        self.constr = self.constraint_function(x, y, self.task)
        if self.task == "multiclass":
            y_probs = self(x)
            y_hat = torch.argmax(y_probs, dim = 1, keepdim=True).double()
            criterion_pred = nn.CrossEntropyLoss(weight=self.class_weights)
            loss_pred = criterion_pred(y_probs.to(torch.float32), y.squeeze().long())
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            logits, y_probs = self(x)
            y_hat = torch.round(y_probs).long()
            class_weights = torch.gather(self.class_weights.repeat(y_hat.size()), dim = 1, index=y_hat)
            criterion_pred = nn.BCEWithLogitsLoss(weight=class_weights)
            loss_pred = criterion_pred(logits, y)
            step_accuracy = self.accuracy(y_probs, y)

        loss =  loss_pred - torch.sum(self.lambdas * self.constr[0])
                
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)             
        opt.step() 

        if self.task == "multiclass":
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            step_accuracy = self.accuracy(y_probs, y)

        fairness =  self.constr[1]
        self.log("train_fairness", torch.max(fairness), logger = True)
        self.log("train_loss", loss, logger = True, prog_bar=True)
        self.log("train_loss_pred", loss_pred, logger = True, prog_bar=True)
        self.log("train_accuracy", step_accuracy, logger = True, prog_bar=True)
        self.log("train_fairness_DE", fairness[0], logger = True)
        self.log("train_fairness_IE", fairness[1], logger = True)
        self.log("train_fairness_SE", fairness[2], logger = True)
        return {"train_fairness": torch.max(fairness), "loss": loss, "loss_pred": loss_pred, "accuracy": step_accuracy}

    def training_epoch_end(self, outputs):    

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_loss_pred = torch.stack([x["loss_pred"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        fairness = torch.stack([x["train_fairness"] for x in outputs]).mean(dim=0)
        self.logger.experiment.add_scalar("Train loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Train accuracy per epoch", avg_acc, self.current_epoch)
        self.logger.experiment.add_scalar("Train fairness per epoch", fairness, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.task == "multiclass":
            y_probs = self(x)
            y_hat = torch.argmax(y_probs, dim = 1, keepdim=True).double()
            criterion_pred = nn.CrossEntropyLoss(weight=self.class_weights)
            loss_pred = criterion_pred(y_probs.to(torch.float32), y.squeeze().long())
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            logits, y_probs = self(x)
            y_hat = torch.round(y_probs).long()
            class_weights = torch.gather(self.class_weights.repeat(y_hat.size()), dim = 1, index=y_hat)
            criterion_pred = nn.BCEWithLogitsLoss(weight=class_weights)
            loss_pred = criterion_pred(logits, y)
            step_accuracy = self.accuracy(y_probs, y)

        self.constr = self.constraint_function(x, y, self.task)
        loss = torch.tensor(loss_pred, requires_grad=True) 
        
        fairness = self.constr[1]
        self.log("val_loss", loss, logger = True)
        self.log("val_loss_pred", loss_pred, logger = True)
        self.log("val_accuracy", step_accuracy, logger = True)
        self.log("val_fairness_DE", fairness[0], logger = True)
        self.log("val_fairness_IE", fairness[1], logger = True)
        self.log("val_fairness_SE", fairness[2], logger = True)
        self.log("val_fairness", torch.max(fairness), logger = True)
        return {"val_loss": loss, "loss_pred": loss_pred, "accuracy": step_accuracy, "val_fairness": torch.max(fairness)}

    def validation_epoch_end(self, outputs):   
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        avg_fairness = torch.stack([x["val_fairness"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Validation loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Validation accuracy per epoch", avg_acc, self.current_epoch)
        self.logger.experiment.add_scalar("Validation fairness per epoch", avg_fairness, self.current_epoch)
        return{"val_loss": avg_loss}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.task == "multiclass":
            y_probs = self(x)
            y_hat = torch.argmax(y_probs, dim = 1, keepdim=True).double()
            criterion_pred = nn.CrossEntropyLoss(weight=self.class_weights)
            loss_pred = criterion_pred(y_probs.to(torch.float32), y.squeeze().long())
            step_accuracy = self.accuracy(y_hat, y.to(torch.int))
        else:
            logits, y_probs = self(x)
            y_hat = torch.round(y_probs).long()
            class_weights = torch.gather(self.class_weights.repeat(y_hat.size()), dim = 1, index=y_hat)
            criterion_pred = nn.BCEWithLogitsLoss(weight=class_weights)
            loss_pred = criterion_pred(logits, y)
            step_accuracy = self.accuracy(y_probs, y)

        self.constr = self.constraint_function(x, y, self.task)
        loss_fairness = torch.sum(torch.tensor(self.lambdas) * self.constr[0])
        loss = loss_pred #- loss_fairness

        fairness = self.constr[1]
        self.log("test_loss", loss, logger = True, prog_bar=True)
        self.log("test_loss_pred", loss_pred, logger = True, prog_bar=True)
        self.log("test_accuracy", step_accuracy, logger = True, prog_bar=True)
        self.log("test_fairness_DE", fairness[0], logger = True)
        self.log("test_fairness_IE", fairness[1], logger = True)
        self.log("test_fairness_SE", fairness[2], logger = True)
        return {"loss": loss, "loss_pred": loss_pred, "accuracy": step_accuracy, "fairness": fairness}


    # ---------------------------------------------------------------------Regressor-------------------------------------------------------------------------------------------



class StandardRegressor(pl.LightningModule):                                      
    def __init__(self, 
                 n_features, 
                 n_classes,
                 hidden_dim,
                 learning_rate,
                 criterion,
                 dropout):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.criterion = criterion()
        self.learning_rate = learning_rate      
        self.save_hyperparameters()

        self.net = nn.Sequential(
            nn.BatchNorm1d(self.n_features),
            nn.Linear(self.n_features, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, n_classes)
            ) 
        
    def forward(self, x):
        out = self.net(x)
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, logger = True, prog_bar=False)
        return {"loss": loss}

    def training_epoch_end(self, outputs):    
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Train loss per epoch", avg_loss, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, logger = True, prog_bar=False)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):        
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Validation loss per epoch", avg_loss, self.current_epoch)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, logger = True, prog_bar=False)
        return {"loss": loss}



class FairRegressor(pl.LightningModule):                                      
    def __init__(self, 
                 n_features, 
                 n_classes,
                 train_data, 
                 batch_size,
                 hidden_dim,
                 learning_rate,
                 sensitive_attributes, #lists all values of A
                 constraints, # of form (gamma_DE, gamma_IE, gamma_SE)
                 criterion_pred,
                 nested_epochs,
                 checkpointpath_density,
                 sensitivity_param,
                 dropout, 
                 column_names):
        super().__init__()
        self.n_features = n_features
        self.n_classes = np.int(n_classes)
        self.train_data = train_data
        self.batch_size = batch_size
        self.criterion_pred = nn.MSELoss()
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.sensitive_attributes = sensitive_attributes
        self.sensitivity_param = sensitivity_param
        self.checkpointpath_density = checkpointpath_density
        self.constraints = torch.tensor(constraints, requires_grad=True)
        self.column_names = column_names
        self.nested_epochs = nested_epochs
        # replace by chosen lagrangian parameters
        self.lambdas = torch.tensor([3.0, 3.0, 3.0], requires_grad=True)
        self.lambdas_old = self.lambdas
        self.mu = 1.5
        self.update_mu = 1.5
        self.save_hyperparameters()

        self.automatic_optimization = False

        self.net = nn.Sequential(
            nn.Linear(self.n_features, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, n_classes)
            ) 

    def forward(self, x):
        out = self.net(x)
        return out
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
    
    def constraint_function(self):

        density_estimator = Density_estimator.load_from_checkpoint(self.checkpointpath_density).eval()
 
        ctf_effects = functions.Ctf_effects_regression_torch(network=self.forward, full_data = self.train_data, sensitivity_param=self.sensitivity_param, density_estimator=density_estimator)
  
        ub_DE_y, lb_DE_y = ctf_effects.DE_binary(self.sensitive_attributes[0], self.sensitive_attributes[1])
        ub_IE_y, lb_IE_y = ctf_effects.IE_binary(self.sensitive_attributes[0], self.sensitive_attributes[1])
        ub_SE_y, lb_SE_y = ctf_effects.SE_binary(self.sensitive_attributes[0], self.sensitive_attributes[1])
        bounds = torch.tensor([ub_DE_y, lb_DE_y, ub_IE_y, lb_IE_y, ub_SE_y, lb_SE_y], requires_grad=True)
        
        bounds = torch.abs(bounds)
        bound_DE = torch.max(bounds[0:2]).requires_grad_()
        bound_IE = torch.max(bounds[2:4]).requires_grad_()
        bound_SE = torch.max(bounds[4:6]).requires_grad_()

        max_bounds = torch.tensor([bound_DE, bound_IE, bound_SE], requires_grad=True)
        return (self.constraints - max_bounds).requires_grad_(), max_bounds

    def training_step(self, batch, batch_idx):    
        x, y = batch
        y_hat = self(x)
        self.constr = self.constraint_function()
        loss_pred = self.criterion_pred(y_hat, y)
        loss = ((loss_pred - torch.sum(self.lambdas * self.constr[0])) + self.mu/2 * torch.sum(self.lambdas -self.lambdas_old)**2).requires_grad_()

        opt = self.optimizers()
        opt.zero_grad()  
        self.manual_backward(loss)
        opt.step()

        fairness =  self.constr[1]    
        self.log("train_loss", loss, logger = True, prog_bar=True)
        self.log("train_loss_pred", loss_pred, logger = True, prog_bar=True)
        self.log("train_fairness_DE", fairness[0], logger = True)
        self.log("train_fairness_IE", fairness[1], logger = True)
        self.log("train_fairness_SE", fairness[2], logger = True)
        self.log("train_fairness", torch.max(fairness), logger = True)
        return {"loss": loss, "loss_pred": loss_pred, "train_fairness": torch.max(fairness)}

    def training_epoch_end(self, outputs):                   
        if self.current_epoch % self.nested_epochs == 0 and self.current_epoch != 0:        
            with torch.no_grad():
                self.lambdas_old = self.lambdas
                self.lambdas.data = np.maximum(self.lambdas.data - self.mu * self.constr[0].detach().numpy(), np.zeros(len(self.lambdas.data)))   
                self.mu = self.update_mu*self.mu

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        fairness = torch.stack([x["train_fairness"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Train loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Train fairness per epoch", fairness, self.current_epoch)

    def validation_step(self, batch, batch_idx):       
        x, y = batch
        y_hat = self(x)
        self.constr = self.constraint_function()
        loss_pred = self.criterion_pred(y_hat, y)
        loss = ((loss_pred - torch.sum(self.lambdas * self.constr[0])))
        fairness =  self.constr[1]
        self.log("val_loss", loss, logger = True, prog_bar=True)
        self.log("val_loss_pred", loss_pred, logger = True, prog_bar=True)
        self.log("val_fairness_DE", fairness[0], logger = True)
        self.log("val_fairness_IE", fairness[1], logger = True)
        self.log("val_fairness_SE", fairness[2], logger = True)
        self.log("val_fairness", torch.max(fairness), logger = True)
        return {"val_loss": loss, "loss_pred": loss_pred, "val_fairness": torch.max(fairness)}

    def validation_epoch_end(self, outputs):   
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_fairness = torch.stack([x["val_fairness"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Validation loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Validation fairness per epoch", avg_fairness, self.current_epoch)
        return{"val_loss": avg_loss, "val_fairness": avg_fairness}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.constr = self.constraint_function()
        loss_pred = self.criterion_pred(y_hat, y)
        loss = torch.tensor((loss_pred - torch.sum(self.lambdas * self.constr[0])), requires_grad=True)

        fairness =  self.constr[1]
        self.log("test_loss", loss, logger = True, prog_bar=True)
        self.log("test_loss_pred", loss_pred, logger = True, prog_bar=True)
        self.log("test_fairness_DE", fairness[0], logger = True)
        self.log("test_fairness_IE", fairness[1], logger = True)
        self.log("test_fairness_SE", fairness[2], logger = True)
        self.log("test_fairness", torch.sum(fairness), logger = True)
        return {"loss": loss, "loss_pred": loss_pred, "test_fairness": torch.sum(fairness)}