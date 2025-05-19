import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import time
import numpy as np

# Parameters
batch_size = 64
subset_fraction = 0.1  # 10% subset

# Prepare MNIST dataset (full training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Flatten MNIST images to vectors for distance calculation
full_data = full_train_dataset.data.view(len(full_train_dataset), -1).float()
full_data = (full_data - full_data.mean()) / full_data.std()  # Normalize features

# Number of samples to select
subset_size = int(len(full_train_dataset) * subset_fraction)

def k_center_greedy(X, n_samples):
    """ 
    Select n_samples indices from X using k-Center Greedy algorithm
    X: torch.Tensor (num_samples x features)
    Returns list of selected indices
    """
    selected = [0]  # Start with first point
    distances = torch.cdist(X[0].unsqueeze(0), X).squeeze(0)  # distances to first point
    
    for _ in range(1, n_samples):
        # Pick the point with the maximum distance to current selected set
        idx = torch.argmax(distances).item()
        selected.append(idx)
        dist_new = torch.cdist(X[idx].unsqueeze(0), X).squeeze(0)
        distances = torch.minimum(distances, dist_new)
    return selected

print("Running k-Center Greedy selection... This may take a few minutes.")

start = time.time()
subset_indices = k_center_greedy(full_data, subset_size)
print(f"Selected {len(subset_indices)} samples in {time.time()-start:.2f} seconds")

subset_train_dataset = Subset(full_train_dataset, subset_indices)
subset_train_loader = DataLoader(subset_train_dataset, batch_size=batch_size, shuffle=True)

print(f"Full training dataset size: {len(full_train_dataset)}")
print(f"k-Center Greedy subset training dataset size: {len(subset_train_dataset)}")

# Model definition (same as before, or import your model)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

num_epochs = 5

# Training loop with logging
for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    start_epoch = time.time()
    for batch_idx, (data, target) in enumerate(subset_train_loader, 1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if batch_idx % 100 == 0 or batch_idx == len(subset_train_loader):
            avg_loss = running_loss / batch_idx
            elapsed = time.time() - start_epoch
            batches_left = len(subset_train_loader) - batch_idx
            eta = elapsed / batch_idx * batches_left
            print(f"Epoch {epoch} [{batch_idx}/{len(subset_train_loader)}] - Avg Loss: {avg_loss:.4f} - Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s")
    
    print(f"Epoch {epoch} completed in {time.time() - start_epoch:.2f}s")

# Evaluate on full test set
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)

print(f"Test Accuracy on full test set: {100 * correct / total:.2f}%")
