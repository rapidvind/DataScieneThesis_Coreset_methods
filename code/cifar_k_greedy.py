import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import time

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
num_epochs = 5
learning_rate = 0.001
subset_fraction = 0.25

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# CIFAR-10 Dataset
full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Flatten loader for selection
flat_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=False)
X = []
for x, _ in flat_loader:
    X.append(x)
X = torch.cat(X, dim=0).view(len(full_train_dataset), -1)

def k_center_greedy(X, subset_size):
    selected = [0]
    distances = torch.cdist(X, X[[0]], p=2).squeeze()

    for _ in tqdm(range(1, subset_size)):
        new_idx = torch.argmax(distances).item()
        selected.append(new_idx)
        new_distances = torch.cdist(X, X[[new_idx]], p=2).squeeze()
        distances = torch.minimum(distances, new_distances)

    return selected

# k-Center Greedy Selection
subset_size = int(subset_fraction * len(full_train_dataset))
print("Running k-Center Greedy selection... This may take a few minutes.")
start_time = time.time()
selected_indices = k_center_greedy(X, subset_size)
elapsed = time.time() - start_time
print(f"Selected {subset_size} samples in {elapsed:.2f} seconds")

subset_train_dataset = Subset(full_train_dataset, selected_indices)
train_loader = DataLoader(subset_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Full training dataset size: {len(full_train_dataset)}")
print(f"k-Center Greedy subset training dataset size: {len(subset_train_dataset)}")
print(f"Training on device: {device}")

# CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    elapsed = time.time() - start_time
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch} [{len(train_loader)}/{len(train_loader)}] - Avg Loss: {avg_loss:.4f} - Elapsed: {elapsed:.2f}s")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

accuracy = 100.0 * correct / total
print(f"Test Accuracy on full CIFAR-10 test set: {accuracy:.2f}%")
