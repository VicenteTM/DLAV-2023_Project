
import torch
from torch import nn
import math
torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection
import numpy as np
#from torch_sparse import coalesce

segmts = [  (0,1), (0,2), (1,2), (0,3), (1,4), (3,4),
            (5,9), (6,10), (7,11), (8,12),
            (9,13), (10,14), (11,15), (12,16)]
edge_index = []
for segm in segmts:
    edge_index.append([segm[0], segm[1]])
    edge_index.append([segm[1], segm[0]])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

class MonolocoModel(nn.Module):
    """
    Architecture inspired by https://github.com/una-dinosauria/3d-pose-baseline
    Pytorch implementation from: https://github.com/weigq/3d_pose_baseline_pytorch
    """

    def __init__(self, input_size, output_size=2, linear_size=256, p_dropout=0.2, num_stage=3):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage

        # process input to linear size
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        self.linear_stages = []
        for _ in range(num_stage):
            self.linear_stages.append(MyLinear(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # post processing
        self.w2 = nn.Linear(self.linear_size, self.output_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        # pre-processing
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        # linear layers
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)
        y = self.w2(y)
        return y


class MyLinear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super().__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):

        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out

from torch_geometric.nn import GCNConv

class KeypointGNN(nn.Module):
    def __init__(self, num_keypoints, num_features, hidden_size=34, num_classes=2,device='cuda'):
        super(KeypointGNN, self).__init__()

        self.num_keypoints = num_keypoints
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.device = device

        # Graph convolution layers
        self.conv1 = GCNConv(self.num_features, self.hidden_size)
        self.conv2 = GCNConv(self.hidden_size, self.hidden_size)
        self.conv3 = GCNConv(self.hidden_size, self.hidden_size)

        # Output layer
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x, edge_index):
        # x: input keypoints [num_keypoints, num_features]
        # edge_index: graph connectivity [2, num_edges]

        # Graph convolution layers
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)

        # Global pooling (e.g., mean pooling) to aggregate node features into a graph-level feature vector in 2D
        #x = torch.mean(x, dim=0)

        # Output layer
        x = self.fc(x)

        return x



class LocoModel(nn.Module):
    def __init__(self, input_size, output_size=2, linear_size=512, p_dropout=0.2, num_stage=3,device='cuda'):
        super().__init__()

        self.num_stage = num_stage
        self.stereo_size = input_size
        self.mono_size = int(input_size / 2)
        self.output_size = output_size - 1
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.linear_stages = []
        self.device = device


        #graph convolution layers
        self.gc1 = KeypointGNN(self.stereo_size, self.stereo_size, self.linear_size, self.linear_size)

        # Preprocessing
        self.w1 = nn.Linear(self.stereo_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

        # Transformer layer
        self.transformer = TransformerModel(self.stereo_size, self.linear_size, num_layers=4, num_heads=34, dropout=self.p_dropout)
        self.batch_norm2 = nn.BatchNorm1d(self.stereo_size)
        self.transformer2 = TransformerModel(self.linear_size, self.linear_size, num_layers=2, num_heads=2, dropout=self.p_dropout)

        # Internal loop
        for _ in range(num_stage):
            self.linear_stages.append(MyLinearSimple(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # Post processing
        self.w2 = nn.Linear(self.linear_size, self.linear_size)
        self.w3 = nn.Linear(self.linear_size, self.linear_size)
        self.batch_norm3 = nn.BatchNorm1d(self.linear_size)

        # Auxiliary task
        self.w_aux = nn.Linear(self.linear_size, 1)

        # Final layers
        self.w_fin = nn.Linear(self.linear_size, self.output_size)

    def forward(self, x):
        print("x",x)
        self.connectivity_graph = generate_connectivity_graph(x.size(0))
        y = self.transformer(x)  # Pass y as both source and target\
        y = y.float()
        y = self.gc1(y.to(self.device), edge_index)
        #y = self.w1(x)
        #y = self.batch_norm1(y)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        # Auxiliary task
        aux = self.w_aux(y)

        # Final layers
        y = self.w2(y)
        y = self.w3(y)
        y = self.batch_norm3(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.w_fin(y)

        # Cat with auxiliary task
        y = torch.cat((y, aux), dim=1)
        return y


class MyLinearSimple(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super().__init__()
        self.l_size = linear_size
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y
        return out


class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout):
        super(TransformerModel, self).__init__()

        # Positional encoding
        self.positional_encoding = PositionalEncoding(input_size)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=input_size, nhead=num_heads, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dim_feedforward=hidden_size, dropout=dropout
        )

    def forward(self, x):
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)
        output = self.transformer(x, x)
        output = output.permute(1, 0, 2)
        output = output.squeeze()
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, input_size, max_length=1594):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_size, 2) * (-math.log(10000.0) / input_size))
        pe = torch.zeros(max_length, input_size)
        pe[:, 0::2] = torch.sin(position * div_term[:input_size // 2])
        pe[:, 1::2] = torch.cos(position * div_term[:input_size // 2])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_length, input_size = x.size()
        pe = self.pe[:, :seq_length, :input_size]
        x = x + pe
        return x
    

