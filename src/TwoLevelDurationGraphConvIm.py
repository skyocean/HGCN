import torch
import pandas as pd

from torch_geometric.data import Data
import torch.optim as optim
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool, global_max_pool, global_add_pool
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.nn as nn
from torch.utils.data import Dataset

import optuna 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from itertools import combinations
import os
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix

# Data preparation
def prepare_data(event_encode, scaled_time_diffs, datatype):
    data_list = []
    for i in range(len(event_encode)):
        node_features = torch.tensor(event_encode[i], dtype=datatype)
        num_events = (node_features[:, 0] != -1).sum()
        edge_index = torch.tensor([[j, j+1] for j in range(num_events-1)], dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(scaled_time_diffs[i][:num_events-1], dtype=torch.float).view(-1, 1)
        graph_data = Data(x=node_features[:num_events], edge_index=edge_index, edge_attr=edge_attr)
        graph_data.num_nodes = num_events
        data_list.append(graph_data)
    return data_list

class CustomDataset(Dataset):
    def __init__(self, event_features, duration_embedding, sequence_features, y):
        self.event_features = event_features
        self.duration_embedding = duration_embedding
        self.sequence_features = sequence_features
        self.y = y

    def __len__(self):
        return len(self.event_features)

    def __getitem__(self, idx):
        return self.event_features[idx], self.duration_embedding[idx],self.sequence_features[idx], self.y[idx]


def train(model, loader, optimizer, criterion, device, l1_lambda):
    model.train()
    total_loss = 0
    correct = 0
    all_labels = []
    all_preds = []
    
    for batch in loader:
        event_data, duration_embedding, sequence_features, labels = batch
        event_data = event_data.to(device)
        duration_embedding = duration_embedding.to(device)
        sequence_features = sequence_features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        output = model(event_data, duration_embedding, sequence_features)
        loss = criterion(output, labels)
        
        # L1 regularization
        l1_penalty = sum(torch.sum(torch.abs(param)) for param in model.parameters())
        loss += l1_lambda * l1_penalty
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        # Append for F1 score calculation
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(pred.cpu().numpy())
       
    accuracy = correct / len(loader.dataset)
    loss = total_loss / len(loader.dataset)
    # Calculate F1-score (macro for multiple classes)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return loss, accuracy, f1

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            event_data, duration_embedding, sequence_features, labels = batch
            event_data = event_data.to(device)
            duration_embedding = duration_embedding.to(device)
            sequence_features = sequence_features.to(device)
            labels = labels.to(device)
            
            output = model(event_data, duration_embedding, sequence_features)
            loss = criterion(output, labels)
            total_loss += loss.item() * labels.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            # Append for F1 score calculation
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())            
    
    accuracy = correct / len(loader.dataset)
    loss = total_loss / len(loader.dataset)
    # Calculate F1-score (weight for imbalance multiple classes )
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return loss, accuracy, f1

class EventSequenceDurationGraphConvModel(nn.Module):
    def __init__(self, num_event_features, num_sequence_features, 
                 gcn_hidden_dims, fc_hidden_dims, output_dim, pooling_method,
                 gcn_batch_norm_flag, fc_batch_norm_flag, gcn_momentum, fc_momentum, gcn_eps, fc_eps, gcn_dropout_flag, fc_dropout_flag, 
                 gcn_dropout_rate, fc_dropout_rate, gcn_activation, fc_activation, gcn_skip_connections, gcn_aggrs,
                 num_duration_features, gcn_hidden_dims_duration, gcn_batch_norm_flag_duration, gcn_momentum_duration, gcn_eps_duration,
                 gcn_dropout_flag_duration, gcn_dropout_rate_duration, gcn_activation_duration, gcn_aggrs_duration,
                 gcn_hidden_dims_concat, gcn_batch_norm_flag_concat, gcn_momentum_concat, gcn_eps_concat,
                 gcn_dropout_flag_concat, gcn_dropout_rate_concat,gcn_activation_concat, gcn_aggrs_concat,
                 fc_hidden_dims_concat, fc_batch_norm_flag_concat, fc_momentum_concat, fc_eps_concat,
                 fc_dropout_flag_concat, fc_dropout_rate_concat, fc_activation_concat):
        super(EventSequenceDurationGraphConvModel, self).__init__()
        
        # Create GCN layers with optional batch normalization, dropout
        self.gcn_layers = nn.ModuleList()
        self.gcn_batch_norms = nn.ModuleList()
        self.gcn_dropouts = nn.ModuleList()
        self.gcn_activation = nn.ModuleList()
        
        in_channels = num_event_features
        for j, hidden_dim in enumerate(gcn_hidden_dims):
            self.gcn_layers.append(GraphConv(in_channels, hidden_dim, aggr=gcn_aggrs[j]))
            if gcn_batch_norm_flag[j]:
                self.gcn_batch_norms.append(nn.BatchNorm1d(hidden_dim, eps=gcn_eps[j], momentum=gcn_momentum[j]))
            else:
                self.gcn_batch_norms.append(None)
            self.gcn_activation.append(self._get_activation(gcn_activation[j]))
            if gcn_dropout_flag[j]:
                self.gcn_dropouts.append(nn.Dropout(gcn_dropout_rate[j]))
            else:
                self.gcn_dropouts.append(None)
            in_channels = hidden_dim

        
        # Create GCN Duration layers with optional batch normalization, dropout
        self.gcn_layers_duration = nn.ModuleList()
        self.gcn_batch_norms_duration = nn.ModuleList()
        self.gcn_dropouts_duration = nn.ModuleList()
        self.gcn_activation_duration = nn.ModuleList()
        in_duration = num_duration_features
        for j, hidden_dim in enumerate(gcn_hidden_dims_duration):
            self.gcn_layers_duration.append(GraphConv(in_duration, hidden_dim, aggr=gcn_aggrs_duration[j]))
            if gcn_batch_norm_flag_duration[j]:
                self.gcn_batch_norms_duration.append(nn.BatchNorm1d(hidden_dim, eps=gcn_eps_duration[j], momentum=gcn_momentum_duration[j]))
            else:
                self.gcn_batch_norms_duration.append(None)
            self.gcn_activation_duration.append(self._get_activation(gcn_activation_duration[j]))
            if gcn_dropout_flag_duration[j]:
                self.gcn_dropouts_duration.append(nn.Dropout(gcn_dropout_rate_duration[j]))
            else:
                self.gcn_dropouts_duration.append(None)
            in_duration = hidden_dim
                

        # Create GCN concate layers with optional batch normalization, dropout
        self.gcn_layers_concat = nn.ModuleList()
        self.gcn_batch_norms_concat = nn.ModuleList()
        self.gcn_dropouts_concat = nn.ModuleList()
        self.gcn_skip_connections = nn.ModuleList()
        self.gcn_activation_concat = nn.ModuleList()
        num_concat_features = in_channels + in_duration
        in_concat = num_concat_features
        for j, hidden_dim in enumerate(gcn_hidden_dims_concat):
            self.gcn_layers_concat.append(GraphConv(in_concat, hidden_dim, aggr=gcn_aggrs_concat[j]))
            self.gcn_skip_connections.append(nn.Linear(in_concat, hidden_dim) if gcn_skip_connections[j] else None)
            if gcn_batch_norm_flag_concat[j]:
                self.gcn_batch_norms_concat.append(nn.BatchNorm1d(hidden_dim, eps=gcn_eps_concat[j], momentum=gcn_momentum_concat[j]))
            else:
                self.gcn_batch_norms_concat.append(None)
            self.gcn_activation_concat.append(self._get_activation(gcn_activation_concat[j]))
            if gcn_dropout_flag_concat[j]:
                self.gcn_dropouts_concat.append(nn.Dropout(gcn_dropout_rate_concat[j]))
            else:
                self.gcn_dropouts_concat.append(None)
            in_concat = hidden_dim
        
        # Create fully connected layers with optional batch normalization and dropout
        self.fc_layers = nn.ModuleList()
        self.fc_batch_norms = nn.ModuleList()
        self.fc_dropouts = nn.ModuleList()
        self.fc_activation = nn.ModuleList()
        in_features = num_sequence_features
        for j, hidden_dim in enumerate(fc_hidden_dims):
            self.fc_layers.append(nn.Linear(in_features, hidden_dim))
            if fc_batch_norm_flag[j]:
                self.fc_batch_norms.append(nn.BatchNorm1d(hidden_dim, eps=fc_eps[j], momentum=fc_momentum[j]))
            else:
                self.fc_batch_norms.append(None)
            self.fc_activation.append(self._get_activation(fc_activation[j]))
            if fc_dropout_flag[j]:
                self.fc_dropouts.append(nn.Dropout(fc_dropout_rate[j]))
            else:
                self.fc_dropouts.append(None)
            in_features = hidden_dim

        # Create fully connected layers for combined features with optional batch normalization and dropout
        self.fc_layers_concat = nn.ModuleList()
        self.fc_batch_norms_concat = nn.ModuleList()
        self.fc_dropouts_concat = nn.ModuleList()
        self.fc_activation_concat = nn.ModuleList()
        in_fc_concat = in_concat + in_features
        for j, hidden_dim in enumerate(fc_hidden_dims_concat):
            self.fc_layers_concat.append(nn.Linear(in_fc_concat, hidden_dim))
            if fc_batch_norm_flag_concat[j]:
                self.fc_batch_norms_concat.append(nn.BatchNorm1d(hidden_dim, eps=fc_eps_concat[j], momentum=fc_momentum_concat[j]))
            else:
                self.fc_batch_norms_concat.append(None)
            self.fc_activation_concat.append(self._get_activation(fc_activation_concat[j]))
            if fc_dropout_flag_concat[j]:
                self.fc_dropouts_concat.append(nn.Dropout(fc_dropout_rate_concat[j]))
            else:
                self.fc_dropouts_concat.append(None)
            in_fc_concat = hidden_dim
        
        # Store pooling method
        self.pooling_method = pooling_method
        
        # Final classification layer
        self.classifier = nn.Linear(in_fc_concat, output_dim)

    def _get_activation(self, activation_name):
        if activation_name == 'relu':
            return nn.ReLU()
        elif activation_name == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation_name == 'elu':
            return nn.ELU()
        elif activation_name == 'tanh':
            return nn.Tanh()
        elif activation_name == 'softplus':
            return nn.Softplus()
        elif activation_name == 'gelu':
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation function: {activation_name}")
    
    def forward(self, data, duration_embedding, sequence_features):
        # Process event-level features through GCN layers with optional batch normalization, dropout, activation
        f = data.x
        # Create a mask for the features; 1 for valid features, 0 for masked features
        mask = (f != -1).float()
        # Apply the mask to the features
        f = f * mask
        
        for i, gcn in enumerate(self.gcn_layers):
            f = gcn(f, data.edge_index, edge_weight=data.edge_attr)
            # Recreate the mask to match the current dimension of x
            mask = (f != -1).float()  # Recalculate mask based on the new x        
            # Apply the mask again after the GCN operation
            f = f * mask
            if self.gcn_batch_norms[i]:
                f = self.gcn_batch_norms[i](f)
            f = self.gcn_activation[i](f)
            if self.gcn_dropouts[i]:
                f = self.gcn_dropouts[i](f)
            f = f * mask
        
        d = duration_embedding.x            
        for i, gcn in enumerate(self.gcn_layers_duration):
            d = gcn(d, duration_embedding.edge_index, edge_weight=duration_embedding.edge_attr)
            if self.gcn_batch_norms_duration[i]:
                d = self.gcn_batch_norms_duration[i](d)
            d = self.gcn_activation_duration[i](d)
            if self.gcn_dropouts_duration[i]:
                d = self.gcn_dropouts_duration[i](d)

        # Concatenate the outputs
        
        x = torch.cat([f, d], dim=1)

        for i, gcn in enumerate(self.gcn_layers_concat):
            skip = x
            x = gcn(x, data.edge_index, edge_weight=data.edge_attr)
            if self.gcn_skip_connections[i]:
                skip = self.gcn_skip_connections[i](skip)
                x += skip  # Add skip connection
            if self.gcn_batch_norms_concat[i]:
                x = self.gcn_batch_norms_concat[i](x)
            x = self.gcn_activation_concat[i](x)
            if self.gcn_dropouts_concat[i]:
                x = self.gcn_dropouts_concat[i](x)
        
        # Pooling to obtain graph-level representation
        if self.pooling_method == 'mean':
            graph_emb = global_mean_pool(x, data.batch)
        elif self.pooling_method == 'max':
            graph_emb = global_max_pool(x, data.batch)
        elif self.pooling_method == 'add':
            graph_emb = global_add_pool(x, data.batch)
        
        # Process sequence-level features through FC layers with optional batch normalization, dropout, and activation
        seq_out = sequence_features
        for i, fc in enumerate(self.fc_layers):
            seq_out = fc(seq_out)
            if self.fc_batch_norms[i]:
                seq_out = self.fc_batch_norms[i](seq_out)
            seq_out = self.fc_activation[i](seq_out)
            if self.fc_dropouts[i]:
                seq_out = self.fc_dropouts[i](seq_out)
                
        # Combine graph and sequence representations
        seq_out_concat = torch.cat([graph_emb, seq_out], dim=1)
        # Process combine features through FC layers with optional batch normalization, dropout, and activation
        for i, fc in enumerate(self.fc_layers_concat):
            seq_out_concat = fc(seq_out_concat)
            if self.fc_batch_norms_concat[i]:
                seq_out_concat = self.fc_batch_norms_concat[i](seq_out_concat)
            seq_out_concat = self.fc_activation_concat[i](seq_out_concat)
            if self.fc_dropouts_concat[i]:
                seq_out_concat = self.fc_dropouts_concat[i](seq_out_concat)
        
        # Classification
        out = self.classifier(seq_out_concat)
        return F.log_softmax(out, dim=1)


class EarlyStopping:
    def __init__(self, patience, delta=0, mode='max'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_score = None
        self.early_stop_counter = 0

        if mode == 'min':
            self.best_score = np.inf
        elif mode == 'max':
            self.best_score = -np.inf
        else:
            raise ValueError("Mode should be either 'min' or 'max'.")

    def __call__(self, score):
        if self.mode == 'min':
            if score < self.best_score - self.delta:
                self.best_score = score
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
        elif self.mode == 'max':
            if score > self.best_score + self.delta:
                self.best_score = score
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

        return self.early_stop_counter >= self.patience

def objective(trial, model_save_folder, train_dataset, test_dataset, num_event_features, num_duration_features, num_sequence_features, output_dim, patience, epochs, device):

    # Define paths to save the best model
    model_save_path = os.path.join(model_save_folder, f'trial_{trial.number}.pth')
    
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    # Create DataLoader with the suggested batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Hyperparameters to tune
    num_gcn_layers = trial.suggest_int('num_gcn_layers', 1, 5)
    gcn_hidden_dims = [trial.suggest_int(f'gcn_hidden_dim_{i}', 32, 256) for i in range(num_gcn_layers)]

    num_gcn_layers_duration = trial.suggest_int('num_gcn_layers_duration', 1, 5)
    gcn_hidden_dims_duration = [trial.suggest_int(f'gcn_hidden_dim__duration{i}', 32, 256) for i in range(num_gcn_layers_duration)]

    num_gcn_layers_concat = trial.suggest_int('num_gcn_layers_concat', 1, 5)
    gcn_hidden_dims_concat = [trial.suggest_int(f'gcn_hidden_dim__concat{i}', 32, 256) for i in range(num_gcn_layers_concat)]
    
    num_fc_layers = trial.suggest_int('num_fc_layers', 1, 3)
    fc_hidden_dims = [trial.suggest_int(f'fc_hidden_dim_{i}', 32, 256) for i in range(num_fc_layers)]
    
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 0.0, 1e-3)
    
    pooling_method = trial.suggest_categorical('pooling_method', ['mean', 'max', 'add'])
    
    # Flags and rates for batch normalization
    gcn_batch_norm_flag = [trial.suggest_categorical(f'gcn_batch_norm_flag_{i}', [True, False]) for i in range(num_gcn_layers)]
    gcn_momentum = [trial.suggest_float(f'gcn_momentum_{i}', 0.01, 0.99) if gcn_batch_norm_flag[i] else 0 for i in range(num_gcn_layers)]
    gcn_eps = [trial.suggest_float(f'gcn_eps_{i}', 1e-5, 1e-2) if gcn_batch_norm_flag[i] else 0 for i in range(num_gcn_layers)]
    gcn_dropout_flag = [trial.suggest_categorical(f'gcn_dropout_flag_{i}', [True, False]) for i in range(num_gcn_layers)]
    gcn_dropout_rate = [trial.suggest_float(f'gcn_dropout_rate_{i}', 0.1, 0.5) if gcn_dropout_flag[i] else 0 for i in range (num_gcn_layers)]
    gcn_activation = [trial.suggest_categorical(f'gcn_activation_{i}', ['relu', 'leaky_relu', 'elu', 'tanh', 'softplus', 'gelu']) for i in range(num_gcn_layers)]
    gcn_aggrs = [trial.suggest_categorical(f'gcn_aggrs_{i}', ['add', 'mean', 'max']) for i in range(num_gcn_layers)]
    
    fc_batch_norm_flag = [trial.suggest_categorical(f'fc_batch_norm_flag_{i}', [True, False]) for i in range(num_fc_layers)]
    fc_momentum = [trial.suggest_float(f'fc_momentum_{i}', 0.01, 0.99) if fc_batch_norm_flag[i] else 0 for i in range(num_fc_layers)]
    fc_eps = [trial.suggest_float(f'fc_eps_{i}', 1e-5, 1e-2) if fc_batch_norm_flag[i] else 0 for i in range(num_fc_layers)]
    fc_dropout_flag = [trial.suggest_categorical(f'fc_dropout_flag_{i}', [True, False]) for i in range(num_fc_layers)]
    fc_dropout_rate = [trial.suggest_float(f'fc_dropout_rate_{i}', 0.1, 0.5) if fc_dropout_flag[i] else 0 for i in range(num_fc_layers)]
    fc_activation = [trial.suggest_categorical(f'fc_activation_{i}', ['relu', 'leaky_relu', 'elu', 'tanh', 'softplus', 'gelu']) for i in range(num_fc_layers)]

    gcn_batch_norm_flag_duration = [trial.suggest_categorical(f'gcn_batch_norm_flag_duration_{i}', [True, False]) for i in range(num_gcn_layers_duration)]
    gcn_momentum_duration = [trial.suggest_float(f'gcn_momentum_duration_{i}', 0.01, 0.99) if gcn_batch_norm_flag_duration[i] else 0 for i in range(num_gcn_layers_duration)]
    gcn_eps_duration = [trial.suggest_float(f'gcn_eps_duration_{i}', 1e-5, 1e-2) if gcn_batch_norm_flag_duration[i] else 0 for i in range(num_gcn_layers_duration)]
    gcn_dropout_flag_duration = [trial.suggest_categorical(f'gcn_dropout_flag_duration_{i}', [True, False]) for i in range(num_gcn_layers_duration)]
    gcn_dropout_rate_duration = [trial.suggest_float(f'gcn_dropout_rate_duration_{i}', 0.1, 0.5) if gcn_dropout_flag_duration[i] else 0 for i in range (num_gcn_layers_duration)]
    gcn_activation_duration = [trial.suggest_categorical(f'gcn_activation_duration_{i}', ['relu', 'leaky_relu', 'elu', 'tanh', 'softplus', 'gelu']) for i in range(num_gcn_layers_duration)]
    gcn_aggrs_duration = [trial.suggest_categorical(f'gcn_aggrs_duration_{i}', ['add', 'mean', 'max']) for i in range(num_gcn_layers_duration)]
    
    # Flags and rates for batch normalization
    gcn_batch_norm_flag_concat = [trial.suggest_categorical(f'gcn_batch_norm_flag_concat_{i}', [True, False]) for i in range(num_gcn_layers_concat)]
    gcn_momentum_concat = [trial.suggest_float(f'gcn_momentum_concat_{i}', 0.01, 0.99) if gcn_batch_norm_flag_concat[i] else 0 for i in range(num_gcn_layers_concat)]
    gcn_eps_concat = [trial.suggest_float(f'gcn_eps_concat_{i}', 1e-5, 1e-2) if gcn_batch_norm_flag_concat[i] else 0 for i in range(num_gcn_layers_concat)]
    gcn_dropout_flag_concat = [trial.suggest_categorical(f'gcn_dropout_flag_concat_{i}', [True, False]) for i in range(num_gcn_layers_concat)]
    gcn_dropout_rate_concat = [trial.suggest_float(f'gcn_dropout_rate_concat_{i}', 0.1, 0.5) if gcn_dropout_flag_concat[i] else 0 for i in range (num_gcn_layers_concat)]
    gcn_activation_concat = [trial.suggest_categorical(f'gcn_activation_concat_{i}', ['relu', 'leaky_relu', 'elu', 'tanh', 'softplus', 'gelu']) for i in range(num_gcn_layers_concat)]
    gcn_aggrs_concat = [trial.suggest_categorical(f'gcn_aggrs_concat_{i}', ['add', 'mean', 'max']) for i in range(num_gcn_layers_concat)]
    
    num_fc_layers_concat = trial.suggest_int('num_fc_layers_concat', 1, 3)
    fc_hidden_dims_concat = [trial.suggest_int(f'fc_hidden_dim_concat_{i}', 32, 256) for i in range(num_fc_layers_concat)]

    fc_batch_norm_flag_concat = [trial.suggest_categorical(f'fc_batch_norm_flag_concat_{i}', [True, False]) for i in range(num_fc_layers_concat)]
    fc_momentum_concat = [trial.suggest_float(f'fc_momentum_concat_{i}', 0.01, 0.99) if fc_batch_norm_flag_concat[i] else 0 for i in range(num_fc_layers_concat)]
    fc_eps_concat = [trial.suggest_float(f'fc_eps_concat_{i}', 1e-5, 1e-2) if fc_batch_norm_flag_concat[i] else 0 for i in range(num_fc_layers_concat)]
    fc_dropout_flag_concat = [trial.suggest_categorical(f'fc_dropout_flag_concat_{i}', [True, False]) for i in range(num_fc_layers_concat)]
    fc_dropout_rate_concat = [trial.suggest_float(f'fc_dropout_rate_concat_{i}', 0.1, 0.5) if fc_dropout_flag_concat[i] else 0 for i in range(num_fc_layers_concat)]
    fc_activation_concat = [trial.suggest_categorical(f'fc_activation_concat_{i}', ['relu', 'leaky_relu', 'elu', 'tanh', 'softplus', 'gelu']) for i in range(num_fc_layers_concat)]

    # Add skip connections flag
    gcn_skip_connections = [trial.suggest_categorical(f'gcn_skip_connections_{i}', [True, False]) for i in range(num_gcn_layers_concat)]
   

    # L1 regularization strength
    l1_lambda = trial.suggest_float('l1_lambda', 0.0, 1e-3)
    
    # Optimizer type
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    # Additional hyperparameters for Adam and SGD
    if optimizer_name == 'Adam':
        beta1 = trial.suggest_float('beta1', 0.85, 0.99)
        beta2 = trial.suggest_float('beta2', 0.9, 0.999)
    elif optimizer_name == 'SGD':
        momentum = trial.suggest_float('momentum', 0.0, 0.9)   
    elif optimizer_name == 'RMSprop':
        alpha = trial.suggest_float('alpha', 0.0, 0.9) 
        momentum_rms = trial.suggest_float('momentum_rms', 0.9, 0.999) 
        eps_rms = trial.suggest_float('eps_rms', 1e-9, 1e-7)
    
    # Learning rate scheduler
    lr_scheduler_name = trial.suggest_categorical('lr_scheduler', ['StepLR', 'ExponentialLR', 'ReduceLROnPlateau', 'PolynomialLR', 'CosineAnnealingLR', 'CyclicLR', 'OneCycleLR'])
    
    if lr_scheduler_name == 'StepLR':
        step_size = trial.suggest_int('step_size', 1, 50)
        stepLRgamma = trial.suggest_float('stepLRgamma', 0.1, 0.9)
    elif lr_scheduler_name == 'ExponentialLR':
        exLRgamma = trial.suggest_float('exLRgamma', 0.85, 0.99)
    elif lr_scheduler_name == 'ReduceLROnPlateau':
        factor = trial.suggest_float('factor', 0.1, 0.9)
        lr_patience = trial.suggest_int('lr_patience', 1, 50)
        lr_threshold = trial.suggest_float('lr_threshold', 1e-4, 1e-2)
        lr_eps = trial.suggest_float('lr_eps', 1e-8, 1e-4)
    elif lr_scheduler_name == 'PolynomialLR':
        power = trial.suggest_float('power', 0.1, 2.0)
        total_iters = trial.suggest_int('total_iters', 2, 300)
    elif lr_scheduler_name == 'CosineAnnealingLR':
        T_max = trial.suggest_int('T_max', 10, 100)
        eta_min = trial.suggest_float('eta_min', 1e-6, 1e-2)
    elif lr_scheduler_name == 'CyclicLR':
        base_lr = trial.suggest_float('base_lr', 1e-5, 1e-2, log=True)
        max_lr_cy = trial.suggest_float('max_lr_cy', 1e-3, 1e-1, log=True)
        step_size_up = trial.suggest_int('step_size_up', 5, 200)
    elif lr_scheduler_name == 'OneCycleLR':
        max_lr_1 = trial.suggest_float('max_lr_1', 1e-3, 1e-1)
        total_steps = len(train_loader)*1000
        pct_start = trial.suggest_float('pct_start', 0.1, 0.5)    
    # Loss function
    loss_function_name = trial.suggest_categorical('loss_function', ['CrossEntropy', 'MultiMargin'])
    
    # Define the model with skip connections option
    model = EventSequenceDurationGraphConvModel(num_event_features=num_event_features, 
                                  num_sequence_features=num_sequence_features, 
                                  gcn_hidden_dims=gcn_hidden_dims, 
                                  fc_hidden_dims=fc_hidden_dims, 
                                  output_dim=output_dim,
                                  pooling_method=pooling_method,
                                  gcn_batch_norm_flag=gcn_batch_norm_flag,
                                  fc_batch_norm_flag=fc_batch_norm_flag,
                                  gcn_momentum=gcn_momentum,
                                  fc_momentum=fc_momentum,
                                  gcn_eps=gcn_eps,
                                  fc_eps=fc_eps,
                                  gcn_dropout_flag=gcn_dropout_flag,
                                  fc_dropout_flag=fc_dropout_flag,
                                  gcn_dropout_rate=gcn_dropout_rate,
                                  fc_dropout_rate=fc_dropout_rate,
                                  gcn_activation=gcn_activation,
                                  fc_activation=fc_activation,
                                  gcn_skip_connections=gcn_skip_connections,
                                  gcn_aggrs=gcn_aggrs,
                                  num_duration_features=num_duration_features, 
                                  gcn_hidden_dims_duration=gcn_hidden_dims_duration, 
                                  gcn_batch_norm_flag_duration=gcn_batch_norm_flag_duration, 
                                  gcn_momentum_duration=gcn_momentum_duration, 
                                  gcn_eps_duration=gcn_eps_duration,
                                  gcn_dropout_flag_duration=gcn_dropout_flag_duration, 
                                  gcn_dropout_rate_duration=gcn_dropout_rate_duration, 
                                  gcn_activation_duration=gcn_activation_duration,
                                  gcn_aggrs_duration=gcn_aggrs_duration,
                                  gcn_hidden_dims_concat=gcn_hidden_dims_concat, 
                                  gcn_batch_norm_flag_concat=gcn_batch_norm_flag_concat, 
                                  gcn_momentum_concat=gcn_momentum_concat, 
                                  gcn_eps_concat=gcn_eps_concat,
                                  gcn_dropout_flag_concat=gcn_dropout_flag_concat, 
                                  gcn_dropout_rate_concat=gcn_dropout_rate_concat,
                                  gcn_activation_concat=gcn_activation_concat,
                                  gcn_aggrs_concat=gcn_aggrs_concat,
                                  fc_hidden_dims_concat=fc_hidden_dims_concat, 
                                  fc_batch_norm_flag_concat=fc_batch_norm_flag_concat, 
                                  fc_momentum_concat=fc_momentum_concat, 
                                  fc_eps_concat=fc_eps_concat,
                                  fc_dropout_flag_concat=fc_dropout_flag_concat, 
                                  fc_dropout_rate_concat=fc_dropout_rate_concat, 
                                  fc_activation_concat=fc_activation_concat)

    model.to(device)
    
   # Select optimizer based on the trial suggestion
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, alpha=alpha, momentum=momentum_rms, eps=eps_rms)
   
    # Define the learning rate scheduler
    if lr_scheduler_name == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=stepLRgamma)
    elif lr_scheduler_name == 'ExponentialLR':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=exLRgamma)
    elif lr_scheduler_name == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=lr_patience, threshold=lr_threshold, eps=lr_eps)
    elif lr_scheduler_name == 'PolynomialLR':
        scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_iters, power=power)
    elif lr_scheduler_name == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max,  eta_min = eta_min)
    elif lr_scheduler_name == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr_cy, step_size_up=step_size_up)
    elif lr_scheduler_name == 'OneCycleLR':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr_1, total_steps=total_steps, pct_start=pct_start)

    
    # Select loss function based on the trial suggestion
    if loss_function_name == 'CrossEntropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_function_name == 'MultiMargin':
        criterion = torch.nn.MultiMarginLoss()
    
    early_stopping = EarlyStopping(patience, mode='max')  # Set patience and mode for early stopping

    best_accuracy = 0
    best_f1 = 0
    best_loss = float('inf')
    best_std_dev = float('inf')  # Initialize the best standard deviation as infinity
    val_losses = []  # List to store validation losses for each epoch  
    
    # Training loop with early stopping
    for epoch in range(epochs):
        train_loss, train_accuracy, train_f1 = train(model, train_loader, optimizer, criterion, device, l1_lambda)
        test_loss, test_accuracy, test_f1 = evaluate(model, test_loader, criterion, device)
        print(f'Epoch {epoch+1}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}')
        
        # Step the scheduler if it's not ReduceLROnPlateau
        if lr_scheduler_name != 'ReduceLROnPlateau':
            scheduler.step()
        else:
            scheduler.step(test_f1)  # ReduceLROnPlateau requires a metric to monitor
         
        # Report test accuracy to Optuna
        trial.report(test_f1, epoch)
        trial.set_user_attr('test_accuracy', test_accuracy)
        trial.set_user_attr('test_loss', test_loss)
        trial.set_user_attr('test_f1', test_f1)
        
        # Calculate standard deviation of validation losses
        val_losses.append(test_loss)  # Store the validation loss for this epoch
        val_loss_std_dev = np.std(val_losses)
        trial.set_user_attr('loss_dev', val_loss_std_dev)
        
        # Prune trial if not promising
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


        if epoch != 0:
            if (test_f1 > best_f1) or (test_f1 == best_f1 and test_loss < best_loss) or (test_f1 == best_f1 and test_loss == best_loss and val_loss_std_dev < best_std_dev):
                # Save the model
                best_accuracy = test_accuracy
                best_f1 = test_f1
                best_std_dev = val_loss_std_dev
                best_loss = test_loss    
                torch.save({
                    'epoch': epoch,
                    'best_accuracy': best_accuracy,
                    'best_f1': best_f1,
                    'best_loss' : best_loss,
                    'best_std_dev' : best_std_dev,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizer_name': optimizer.__class__.__name__, 
                    'loss_function': criterion, 
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'batch_size': batch_size,
                    'l1_lambda': l1_lambda,
                    'weight_decay': optimizer.param_groups[0]['weight_decay'],
                    # Include optimizer-specific parameters
                    'beta1': beta1 if optimizer_name == 'Adam' else None,
                    'beta2': beta2 if optimizer_name == 'Adam' else None,
                    'momentum': momentum if optimizer_name == 'SGD' else None,
                    'momentum_rms': momentum_rms if optimizer_name == 'RMSprop' else None,
                    'alpha': alpha if optimizer_name == 'RMSprop' else None,
                    'eps_rms': eps_rms if optimizer_name == 'RMSprop' else None,
                    # Include scheduler parameters
                    'lr_scheduler_name': scheduler.__class__.__name__,  
                    'scheduler_state_dict': scheduler.state_dict(),
                    'step_size': step_size if lr_scheduler_name == 'StepLR' else None,
                    'stepLRgamma': stepLRgamma if lr_scheduler_name == 'StepLR' else None,
                    'exLRgamma':exLRgamma if lr_scheduler_name == 'ExponentialLR' else None,
                    'factor': factor if lr_scheduler_name == 'ReduceLROnPlateau' else None,
                    'lr_patience': lr_patience if lr_scheduler_name == 'ReduceLROnPlateau' else None,
                    'lr_threshold': lr_threshold if lr_scheduler_name == 'ReduceLROnPlateau' else None,
                    'lr_eps': lr_eps if lr_scheduler_name == 'ReduceLROnPlateau' else None,
                    'power': power if lr_scheduler_name == 'PolynomialLR' else None,
                    'total_iters': total_iters if lr_scheduler_name == 'PolynomialLR' else None,
                    'T_max': T_max if lr_scheduler_name == 'CosineAnnealingLR' else None,
                    'eta_min': eta_min if lr_scheduler_name == 'CosineAnnealingLR' else None,
                    'max_lr_cy': max_lr_cy if lr_scheduler_name == 'CyclicLR' else None,
                    'base_lr': base_lr if lr_scheduler_name == 'CyclicLR' else None,
                    'step_size_up': step_size_up if lr_scheduler_name == 'CyclicLR' else None,
                    'max_lr_1':  max_lr_1 if lr_scheduler_name == 'OneCycleLR' else None,
                    'total_steps':  total_steps if lr_scheduler_name == 'OneCycleLR' else None,
                    'pct_start':  pct_start if lr_scheduler_name == 'OneCycleLR' else None,
                    # Save model hyperparameters
                    'num_event_features': num_event_features,
                    'num_sequence_features': num_sequence_features,
                    'gcn_hidden_dims': gcn_hidden_dims,
                    'fc_hidden_dims': fc_hidden_dims,
                    'output_dim': output_dim,
                    'pooling_method': pooling_method,
                    'gcn_batch_norm_flag': gcn_batch_norm_flag,
                    'fc_batch_norm_flag': fc_batch_norm_flag,
                    'gcn_momentum': gcn_momentum,
                    'fc_momentum': fc_momentum,
                    'gcn_eps': gcn_eps,
                    'fc_eps': fc_eps,
                    'gcn_dropout_flag': gcn_dropout_flag,
                    'fc_dropout_flag': fc_dropout_flag,
                    'gcn_dropout_rate': gcn_dropout_rate,
                    'fc_dropout_rate': fc_dropout_rate,
                    'gcn_activation': gcn_activation,
                    'gcn_aggrs': gcn_aggrs,
                    'fc_activation': fc_activation,
                    'gcn_skip_connections': gcn_skip_connections,
                    'num_duration_features': num_duration_features, 
                    'gcn_hidden_dims_duration': gcn_hidden_dims_duration, 
                    'gcn_batch_norm_flag_duration':gcn_batch_norm_flag_duration, 
                    'gcn_momentum_duration':gcn_momentum_duration, 
                    'gcn_eps_duration': gcn_eps_duration,
                    'gcn_dropout_flag_duration':gcn_dropout_flag_duration, 
                    'gcn_dropout_rate_duration':gcn_dropout_rate_duration, 
                    'gcn_activation_duration':gcn_activation_duration,
                    'gcn_aggrs_duration': gcn_aggrs_duration,
                    'gcn_hidden_dims_concat':gcn_hidden_dims_concat, 
                    'gcn_batch_norm_flag_concat':gcn_batch_norm_flag_concat, 
                    'gcn_momentum_concat':gcn_momentum_concat, 
                    'gcn_eps_concat': gcn_eps_concat,
                    'gcn_dropout_flag_concat':gcn_dropout_flag_concat, 
                    'gcn_dropout_rate_concat':gcn_dropout_rate_concat,
                    'gcn_activation_concat':gcn_activation_concat,
                    'gcn_aggrs_concat': gcn_aggrs_concat,
                    'fc_hidden_dims_concat':fc_hidden_dims_concat, 
                    'fc_batch_norm_flag_concat':fc_batch_norm_flag_concat, 
                    'fc_momentum_concat':fc_momentum_concat, 
                    'fc_eps_concat': fc_eps_concat,
                    'fc_dropout_flag_concat':fc_dropout_flag_concat, 
                    'fc_dropout_rate_concat':fc_dropout_rate_concat,
                    'fc_activation_concat':fc_activation_concat
                    }, model_save_path)

        # Early stopping
        if early_stopping(test_f1):
            print("Early stopping triggered")
            break
    
    return best_f1

def load_model(model_class, path, device):
    checkpoint = torch.load(path, map_location=device)
    
    # Restore the model with the saved architecture
    model = model_class(
        num_event_features=checkpoint['num_event_features'], 
        num_sequence_features=checkpoint['num_sequence_features'], 
        gcn_hidden_dims=checkpoint['gcn_hidden_dims'], 
        fc_hidden_dims=checkpoint['fc_hidden_dims'], 
        output_dim=checkpoint['output_dim'],
        pooling_method=checkpoint['pooling_method'],
        gcn_batch_norm_flag=checkpoint['gcn_batch_norm_flag'],
        fc_batch_norm_flag=checkpoint['fc_batch_norm_flag'],
        gcn_momentum=checkpoint['gcn_momentum'] if checkpoint['gcn_batch_norm_flag'] else 0,
        fc_momentum=checkpoint['fc_momentum'] if checkpoint['fc_batch_norm_flag']else 0,
        gcn_eps=checkpoint['gcn_eps'] if checkpoint['gcn_batch_norm_flag'] else 0,
        fc_eps=checkpoint['fc_eps'] if checkpoint['fc_batch_norm_flag']else 0,
        gcn_dropout_flag=checkpoint['gcn_dropout_flag'],
        fc_dropout_flag=checkpoint['fc_dropout_flag'],
        gcn_dropout_rate=checkpoint['gcn_dropout_rate'] if checkpoint['gcn_dropout_flag']else 0,
        fc_dropout_rate=checkpoint['fc_dropout_rate'] if checkpoint['fc_dropout_flag'] else 0,
        gcn_activation=checkpoint['gcn_activation'],
        gcn_aggrs=checkpoint['gcn_aggrs'],
        fc_activation=checkpoint['fc_activation'],
        gcn_skip_connections=checkpoint['gcn_skip_connections'],
        num_duration_features=checkpoint['num_duration_features'], 
        gcn_hidden_dims_duration=checkpoint['gcn_hidden_dims_duration'],                                   
        gcn_batch_norm_flag_duration=checkpoint['gcn_batch_norm_flag_duration'], 
        gcn_momentum_duration=checkpoint['gcn_momentum_duration']if checkpoint['gcn_batch_norm_flag_duration'] else 0, 
        gcn_eps_duration=checkpoint['gcn_eps_duration'] if checkpoint['gcn_batch_norm_flag_duration'] else 0,
        gcn_dropout_flag_duration=checkpoint['gcn_dropout_flag_duration'], 
        gcn_dropout_rate_duration=checkpoint['gcn_dropout_rate_duration'] if checkpoint['gcn_dropout_flag_duration'] else 0, 
        gcn_activation_duration=checkpoint['gcn_activation_duration'],
        gcn_aggrs_duration=checkpoint['gcn_aggrs_duration'],
        gcn_hidden_dims_concat=checkpoint['gcn_hidden_dims_concat'], 
        gcn_batch_norm_flag_concat=checkpoint['gcn_batch_norm_flag_concat'] , 
        gcn_momentum_concat=checkpoint['gcn_momentum_concat'] if checkpoint['gcn_batch_norm_flag_concat'] else 0,    
        gcn_eps_concat=checkpoint['gcn_eps_concat'] if checkpoint['gcn_batch_norm_flag_concat'] else 0,
        gcn_dropout_flag_concat=checkpoint['gcn_dropout_flag_concat'], 
        gcn_dropout_rate_concat=checkpoint['gcn_dropout_rate_concat']if checkpoint['gcn_dropout_flag_concat'] else 0,
        gcn_activation_concat=checkpoint['gcn_activation_concat'],
        gcn_aggrs_concat=checkpoint['gcn_aggrs_concat'],
        fc_hidden_dims_concat=checkpoint['fc_hidden_dims_concat'], 
        fc_batch_norm_flag_concat=checkpoint['fc_batch_norm_flag_concat'] , 
        fc_momentum_concat=checkpoint['fc_momentum_concat'] if checkpoint['fc_batch_norm_flag_concat'] else 0,    
        fc_eps_concat=checkpoint['fc_eps_concat'] if checkpoint['fc_batch_norm_flag_concat'] else 0,
        fc_dropout_flag_concat=checkpoint['fc_dropout_flag_concat'], 
        fc_dropout_rate_concat=checkpoint['fc_dropout_rate_concat']if checkpoint['fc_dropout_flag_concat'] else 0,
        fc_activation_concat=checkpoint['fc_activation_concat']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    # Re-initialize the optimizer with the same parameters
    optimizer_name = checkpoint['optimizer_name']
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=checkpoint['learning_rate'], weight_decay=checkpoint['weight_decay'], betas=(checkpoint['beta1'],checkpoint['beta2']))
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=checkpoint['learning_rate'], weight_decay=checkpoint['weight_decay'], momentum=checkpoint['momentum'])
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=checkpoint['learning_rate'], weight_decay=checkpoint['weight_decay'], momentum=checkpoint['momentum_rms'], alpha = checkpoint['alpha'], eps = checkpoint['eps_rms'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Reconstruct the scheduler based on its type
    lr_scheduler_name = checkpoint['lr_scheduler_name']
    if  lr_scheduler_name == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=checkpoint['step_size'], gamma=checkpoint['stepLRgamma'])
    elif lr_scheduler_name == 'ExponentialLR':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=checkpoint['exLRgamma'])
    elif lr_scheduler_name == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=checkpoint['factor'], patience=checkpoint['lr_patience'], threshold=checkpoint['lr_threshold'],eps=checkpoint['lr_eps'])
    elif lr_scheduler_name == 'PolynomialLR':
        scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=checkpoint['total_iters'], power=checkpoint['power'])
    elif lr_scheduler_name == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=checkpoint['T_max'],  eta_min = checkpoint['eta_min'])
    elif lr_scheduler_name == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=checkpoint['base_lr'], max_lr=checkpoint['max_lr_cy'], step_size_up=checkpoint['step_size_up'])
    elif lr_scheduler_name == 'OneCycleLR':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=checkpoint['max_lr_1'], total_steps=checkpoint['total_steps'], pct_start=checkpoint['pct_start'])
    
    # Set the scheduler state
    scheduler.load_state_dict(checkpoint['scheduler_state_dict']) 
    # Load additional parameters
    loss_function = checkpoint['loss_function']
    batch_size = checkpoint['batch_size']
    #learning_rate = checkpoint['learning_rate']
    l1_lambda = checkpoint['l1_lambda']
    #weight_decay = checkpoint['weight_decay']
    best_accuracy = checkpoint['best_accuracy']
    best_loss = checkpoint ['best_loss']      
    best_std_dev = checkpoint['best_std_dev']
    best_f1 = checkpoint['best_f1']
   
    epoch = checkpoint['epoch']
    
    return model, optimizer, loss_function, batch_size, l1_lambda, epoch, best_accuracy, best_loss, best_std_dev, best_f1

def f1_eva(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            event_data, duration_embedding, sequence_features, labels = batch
            event_data = event_data.to(device)
            duration_embedding = duration_embedding.to(device)
            sequence_features = sequence_features.to(device)
            labels = labels.to(device)
            
            output = model(event_data, duration_embedding, sequence_features)
            loss = criterion(output, labels)
            total_loss += loss.item() * labels.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            # Append for F1 score calculation
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())            
    
    accuracy = correct / len(loader.dataset)
    loss = total_loss / len(loader.dataset)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, digits=4)

    return loss, accuracy, conf_matrix, class_report