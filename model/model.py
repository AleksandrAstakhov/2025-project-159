import os
import glob
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

def graph_conv_batch(X, A, W):
    """
    Простейшая GCN-like операция на батче:
      X shape: [B, C, in_dim]
      A shape: [B, C, C]
      W shape: [in_dim, out_dim]
    Вычисляем out = A * (X * W).
    out => [B, C, out_dim].
    """
    XW = torch.matmul(X, W)       # [B, C, out_dim]
    out = torch.matmul(A, XW)     # [B, C, out_dim]
    return out

class DCRNNCellBatch(nn.Module):
    """
    Аналог GRUCell с графовой свёрткой, обрабатывающий батч [B, C, ...].
    """
    def __init__(self, in_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Параметры для z
        self.Wz_x = nn.Parameter(torch.Tensor(in_dim, hidden_dim))
        self.Wz_h = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        
        # Параметры для r
        self.Wr_x = nn.Parameter(torch.Tensor(in_dim, hidden_dim))
        self.Wr_h = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        
        # Параметры для h_tilde
        self.Wh_x = nn.Parameter(torch.Tensor(in_dim, hidden_dim))
        self.Wh_h = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        for w in [self.Wz_x, self.Wz_h, self.Wr_x, self.Wr_h, self.Wh_x, self.Wh_h]:
            nn.init.xavier_uniform_(w)

    def forward(self, x_t, h_prev, A_t):
        """
        x_t: [B, C, in_dim]
        h_prev: [B, C, hidden_dim]
        A_t: [B, C, C]
        """
        # z_t
        z_t = torch.sigmoid(
            graph_conv_batch(x_t, A_t, self.Wz_x) + 
            graph_conv_batch(h_prev, A_t, self.Wz_h)
        )
        
        # r_t
        r_t = torch.sigmoid(
            graph_conv_batch(x_t, A_t, self.Wr_x) +
            graph_conv_batch(h_prev, A_t, self.Wr_h)
        )
        
        # dropout на (r_t * h_prev)
        h_tilde_in = r_t * h_prev
        h_tilde_in = self.dropout(h_tilde_in)
        
        h_tilde = torch.tanh(
            graph_conv_batch(x_t, A_t, self.Wh_x) +
            graph_conv_batch(h_tilde_in, A_t, self.Wh_h)
        )
        
        # h_t
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_t

class DCRNNBatch(nn.Module):
    """
    Многослойная версия (но для L шагов). 
    Мы будем использовать L=1 для SEED-IV (каждый trial - отдельная выборка).
    """
    def __init__(self, in_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList([
            DCRNNCellBatch(
                in_dim=(in_dim if i == 0 else hidden_dim),
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            for i in range(num_layers)
        ])
    
    def forward(self, X_seq, A_seq):
        """
        X_seq: [B, L, C, in_dim]
        A_seq: [B, L, C, C]
        
        Вернём h_all: [B, L, C, hidden_dim] (скрытые состояния последнего слоя).
        Для L=1, это просто [B,1,C,hidden_dim].
        """
        B, L, C, in_dim = X_seq.shape
        
        # Инициализация h (для каждого слоя)
        h = [
            torch.zeros(B, C, self.hidden_dim, device=X_seq.device)
            for _ in range(self.num_layers)
        ]
        
        h_all_steps = []
        
        for t in range(L):
            x_t = X_seq[:, t]  # [B, C, in_dim]
            A_t = A_seq[:, t]  # [B, C, C]
            
            for i, cell in enumerate(self.cells):
                if i == 0:
                    input_i = x_t
                else:
                    input_i = h[i-1]
                
                h[i] = cell(input_i, h[i], A_t)
            
            h_all_steps.append(h[-1].unsqueeze(1))  # [B,1,C,hidden_dim]
        
        h_all = torch.cat(h_all_steps, dim=1)  # [B,L,C,hidden_dim]
        return h_all

class DCRNNClassifierBatch(nn.Module):
    """
    Классификатор на основе DCRNN; 
    каждый пример -> [B, 1, C, in_dim], pooling -> logits.
    """
    def __init__(self, in_dim, hidden_dim, num_classes=4, num_layers=1, dropout=0.0, pooling='mean'):
        super().__init__()
        self.dcrnn = DCRNNBatch(in_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        self.pooling = pooling
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, X_seq, A_seq):
        """
        X_seq: [B, L, C, in_dim]
        A_seq: [B, L, C, C]
        """
        h_all = self.dcrnn(X_seq, A_seq)  # [B, L, C, hidden_dim]
        B, L, C, H = h_all.shape
        
        if self.pooling == 'mean':
            # Усредним по L и C
            h_mean = h_all.mean(dim=2).mean(dim=1)  # -> [B, hidden_dim]
        else:
            # Берём последнее время, усредняем каналы
            h_last = h_all[:, -1]  # [B, C, hidden_dim]
            h_mean = h_last.mean(dim=1)  # [B, hidden_dim]
        
        logits = self.classifier(h_mean)  # [B, num_classes]
        return logits
