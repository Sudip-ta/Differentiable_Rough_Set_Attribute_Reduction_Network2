import torch
import torch.nn as nn
try:
    from torch_geometric.nn import MessagePassing
except ImportError:
    print("Please install torch-geometric: pip install torch-geometric")

class RoughGraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(RoughGraphConv, self).__init__(aggr='add') # "Add" aggregation
        self.lin_low = nn.Linear(in_channels, out_channels)
        self.lin_bnd = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, labels):
        # Step 4.5.1: Rough Granulation Message Passing
        # In a real setup, you'd pre-calculate which neighbors are 'Consistent' 
        # based on Rough Set theory. Here we simplify the logic:
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        # x_j represents neighbor features
        # We apply the transformation from Step 4.5.1
        # h_low = AGGREGATE(consistent neighbors)
        return torch.relu(self.lin_low(x_j))

class GraphEmotionNet(nn.Module):
    def __init__(self, feature_dim):
        super(GraphEmotionNet, self).__init__()
        self.conv1 = RoughGraphConv(feature_dim, 16)
        self.classifier = nn.Linear(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index, None)
        return torch.sigmoid(self.classifier(x))