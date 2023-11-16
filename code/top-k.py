import pandas as pd
import numpy as np
import networkx as nx
from scipy import spatial
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Reading data from a CSV file
data = pd.read_csv('feature.csv')

# Convert the data to a NumPy array
features = data.values[:, :-1]
labels = data.values[:, -1]

# Randomly shuffle the order of the data
random_indices = np.random.permutation(len(features))
features = features[random_indices]
labels = labels[random_indices]

# Compute the cosine similarity matrix
cosine_similarities = spatial.distance.cdist(features, features, 'cosine')
cosine_similarities = 1 - cosine_similarities

# Build a graph by selecting the top 'k' similar nodes for each node based on similarity
k = 12
num_nodes = features.shape[0]
subgraph = nx.Graph()

# Iterate over each node
for i in range(num_nodes):
    # Sort similarity in descending order
    sorted_neighbors = np.argsort(cosine_similarities[i])[::-1]

    # Select top 'k' neighbors
    selected_neighbors = sorted_neighbors[1:k + 1]

    # Add selected edges to the subgraph
    subgraph.add_edges_from([(i, j) for j in selected_neighbors])

features = features.astype(np.float32)
labels = labels.astype(np.int64)

edge_index = torch.tensor(list(subgraph.edges)).t().contiguous()
x = torch.tensor(features, dtype=torch.float)
y = torch.tensor(labels, dtype=torch.long)
data = Data(x=x, edge_index=edge_index, y=y)

# Splitting the dataset into training and testing sets
indices = torch.arange(num_nodes)
train_indices, test_indices = train_test_split(indices, test_size=0.2)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_indices] = 1
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask[test_indices] = 1
data.train_mask = train_mask
data.test_mask = test_mask

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, 16)
        self.conv2 = SAGEConv(16, 32)
        self.fc1 = torch.nn.Linear(32, 64)
        self.fc2 = torch.nn.Linear(64, 128)
        self.fc3 = torch.nn.Linear(128, num_classes)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Training model
model = GraphSAGE(num_features=features.shape[1], num_classes=len(np.unique(labels)))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    target = data.y[data.train_mask] + 1
    target = target.clamp(min=0, max=len(np.unique(labels)) - 1)
    loss = criterion(out[data.train_mask], target)
    loss.backward()
    optimizer.step()
    print('\r {} / 200'.format(epoch + 1), end='')

# Testing model
model.eval()
with torch.no_grad():
    test_out = model(data.x, data.edge_index)
    test_pred = test_out.argmax(dim=1)[data.test_mask]
    test_true = data.y[data.test_mask] + 1
    test_true = test_true.clamp(min=0, max=len(np.unique(labels)) - 1)
    test_pred = test_pred.cpu().numpy()
    test_true = test_true.cpu().numpy()

    precision = precision_score(test_true, test_pred)
    recall = recall_score(test_true, test_pred)
    f1 = f1_score(test_true, test_pred)
    test_acc = (test_pred == test_true).sum().item() / test_true.shape[0]
    print(f'precision: {precision:.4f}')
    print(f'recall: {recall:.4f}')
    print(f'F1 score: {f1:.4f}')
    print(f'Testing accuracy: {test_acc:.4f}')