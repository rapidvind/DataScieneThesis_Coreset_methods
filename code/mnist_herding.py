import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import time

# Parameters
batch_size = 64
subset_fraction = 0.1

# Load full MNIST training set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Flatten and normalize image data for feature space
full_data = full_train_dataset.data.view(len(full_train_dataset), -1).float()
full_data = (full_data - full_data.mean()) / full_data.std()

subset_size = int(len(full_train_dataset) * subset_fraction)

def herding_selection(X, n_samples):
    """
    Herding: Select n_samples points to approximate the mean of X
    """
    mean_feature = X.mean(dim=0)
    selected_indices = []
    current_sum = torch.zeros_like(mean_feature)

    for _ in range(n_samples):
        residual = mean_feature - current_sum / (len(selected_indices) + 1 if selected_indices else 1)
        distances = torch.mv(X, residual)
        best_idx = torch.argmax(distances).item()
        selected_indices.append(best_idx)
        current_sum += X[best_idx]
    return selected_indices

print("Running Herding selection...")
start = time.time()
subset_indices = herding_selection(full_data, subset_size)
print(f"Selected {len(subset_indices)} samples in {time.time() - start:.2f} seconds")

subset_train_dataset = Subset(full_train_dataset, subset_indices)
subset_train_loader = DataLoader(subset_train_dataset, batch_size=batch_size, shuffle=True)

print(f"Full training dataset size: {len(full_train_dataset)}")
print(f"Herding subset training dataset size: {len(subset_train_dataset)}")

# Model definition (same CNN)
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

# Training loop with logs
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

# Evaluation on full test set
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
