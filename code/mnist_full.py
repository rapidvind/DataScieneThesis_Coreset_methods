import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Model with dynamic flatten size calculation
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU()
        )
        # Compute flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 28, 28)
            dummy = self.conv_layers(dummy)
            flatten_size = dummy.numel()

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_size, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Data transforms and loaders
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_data = datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = DataLoader(test_data, batch_size=1000)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

num_epochs = 5
total_batches = len(train_loader)

start_time = time.time()

for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_start = time.time()
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 100 == 0 or batch_idx == total_batches:
            avg_loss = running_loss / batch_idx
            elapsed = time.time() - epoch_start
            batches_left = total_batches - batch_idx
            est_remaining = (elapsed / batch_idx) * batches_left
            print(f"Epoch {epoch} [{batch_idx}/{total_batches}] - "
                  f"Avg Loss: {avg_loss:.4f} - "
                  f"Elapsed: {elapsed:.1f}s - "
                  f"ETA: {est_remaining:.1f}s")

    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch} completed in {epoch_time:.2f}s")

total_time = time.time() - start_time

# Evaluation
model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

accuracy = 100. * correct / len(test_loader.dataset)
print(f"\nFull Dataset Accuracy: {accuracy:.2f}%")
print(f"Total training time: {total_time / 60:.2f} minutes")
