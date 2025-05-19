import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Parameters
batch_size = 64
subset_fraction = 0.1  # 10% subset
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Random subset indices
subset_size = int(len(full_train_dataset) * subset_fraction)
subset_indices = torch.randperm(len(full_train_dataset))[:subset_size]
subset_train_dataset = Subset(full_train_dataset, subset_indices)

# DataLoaders
train_loader = DataLoader(subset_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Full training dataset size: {len(full_train_dataset)}")
print(f"Random subset training dataset size: {len(subset_train_dataset)}")

# Define a simple CNN model (similar to before)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.log_softmax(x, dim=1)
        return output

model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.NLLLoss()

# Training loop with logging
for epoch in range(1, epochs + 1):
    model.train()
    epoch_loss = 0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        if batch_idx % 100 == 0 or batch_idx == len(train_loader):
            avg_loss = epoch_loss / batch_idx
            elapsed = time.time() - start_time
            eta = elapsed / batch_idx * (len(train_loader) - batch_idx)
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] - Avg Loss: {avg_loss:.4f} - Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s")

    print(f"Epoch {epoch} completed in {time.time() - start_time:.2f}s")

# Evaluation on full test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

accuracy = 100.0 * correct / total
print(f"Test Accuracy on full test set: {accuracy:.2f}%")
