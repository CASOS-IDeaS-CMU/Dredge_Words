from torch_geometric.nn.conv import transformer_conv, GCNConv
import torch.nn.functional as F
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU
import torch
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, global_add_pool, SAGEConv, GATConv, SAGPooling, Linear
from torch_geometric.nn import global_mean_pool, global_max_pool
ATTN_HEADS = 1

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, heads = ATTN_HEADS)
        self.conv2 = GATConv((-1, -1), 128, heads = ATTN_HEADS)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc1 = Linear(128*ATTN_HEADS, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.fc2 = Linear(64, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.log_softmax(dim=-1)



class HetGAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, heads = ATTN_HEADS)
        self.conv2 = GATConv((-1, -1), 128, heads = ATTN_HEADS)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc1 = Linear(128*ATTN_HEADS, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.fc2 = Linear(64, out_channels)
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in ['websites', 'users']:
            self.lin_dict[node_type] = Linear(-1, 256)
            
    def forward(self, x, edge_index):
        for node_type, x in x.items():
            x[node_type] = self.dropout(self.lin_dict[node_type](x).relu())
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.log_softmax(dim=-1)


class HomoGAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # heads affect dimensions of second layer
        self.gc1 = GATConv((-1, -1),hidden_channels, heads = ATTN_HEADS)
        self.gc2 = GATConv((-1, -1), 1024, heads = ATTN_HEADS)
        self.fc1 = torch.nn.Linear(1024*ATTN_HEADS, 4092)
        self.bn1 = torch.nn.BatchNorm1d(4092)
        self.fc2 = torch.nn.Linear(4092, 1024)
        self.bn2 = torch.nn.BatchNorm1d(1024)
        self.fc3 = torch.nn.Linear(1024, 256)
        self.bn3 = torch.nn.BatchNorm1d(256)
        self.fc4 = torch.nn.Linear(256, out_channels)
        self.dropout = torch.nn.Dropout(p=0.6)
    def forward(self, x, edge_index):
        x = self.gc1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.gc2(x, edge_index).relu()
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        x = x.log_softmax(dim=-1)
        return x


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
        self.dropout = torch.nn.Dropout(p=0.5)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=-1)


class HetGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in ['websites', 'users']:
            self.lin_dict[node_type] = Linear(-1, 256)
        
    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.dropout(self.lin_dict[node_type](x).relu())
        x = self.conv1(x_dict, edge_index_dict).relu()
        x = self.dropout(x_dict)
        x = self.conv2(x_dict, edge_index_dict)
        return x_dict.log_softmax(dim=-1)

class convGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv((-1, -1), hidden_channels)
        self.conv2 = GCNConv((-1, -1), out_channels)
        self.dropout = torch.nn.Dropout(p=0.5)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x.log_softmax(dim=-1)


class dumb_GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(23, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 64)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear = torch.nn.Linear(64, out_channels)
    
    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.dropout(x)
        x = self.linear(x)
        return x.log_softmax(dim=-1)
    
class dumbest_GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return x.log_softmax(dim=-1)
    
from torch_geometric.nn.conv import transformer_conv, GCNConv
import torch.nn.functional as F
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear
import torch
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, global_add_pool, SAGEConv, GATConv, SAGPooling, GraphConv, GCN2Conv, GATv2Conv, GeneralConv, PDNConv, GMMConv 
from torch_geometric.nn import global_mean_pool, global_max_pool

class GNN_v2(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, use_weights = True):
        super().__init__()
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 64)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear = torch.nn.Linear(64, out_channels)
        self.use_weights = use_weights
    
    def forward(self, x, edge_index, edge_weight):
        if self.use_weights:
            x = self.conv1(x, edge_index, edge_weight).relu()
            x = self.conv2(x, edge_index, edge_weight).relu()
        else:
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        x = self.linear(x)
        return x.log_softmax(dim=-1)
    
class GNN_v1(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return x.log_softmax(dim=-1)


from torch_geometric.nn import HGTConv, Linear


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['websites']).log_softmax(dim=-1)

