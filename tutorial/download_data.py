import torchvision
import torchvision.transforms as transforms
import os

# Define the scratch path
scratch_dir = os.environ.get("SCRATCH", "/scratch/munro")  # Fallback to /scratch/munro
dataset_path = os.path.join(scratch_dir, "datasets", "MNIST")

# Ensure directory exists
os.makedirs(dataset_path, exist_ok=True)

# Download MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root=dataset_path, train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root=dataset_path, train=False, download=True, transform=transform)

print(f"MNIST dataset downloaded to: {dataset_path}")
