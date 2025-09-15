import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

# Set the directory for datasets and checkpoints using SCRATCH
scratch_dir = os.environ.get("SCRATCH", "/scratch/munro")  # Fallback to /scratch/munro if SCRATCH is not set
dataset_path = os.path.join(scratch_dir, "datasets", "MNIST")
checkpoint_dir = os.path.join(scratch_dir, "checkpoints")

# Ensure that the directories exist
os.makedirs(dataset_path, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a simple MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

# Load MNIST dataset from $SCRATCH directory
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root=dataset_path, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Initialize model, loss function, and optimizer
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 2  # Run only 2 epochs for testing
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")

    # Save checkpoint after each epoch
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

print("Training complete.")
