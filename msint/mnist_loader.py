import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import random

# Define the transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    # Remove normalization for now to debug
    # transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
mnist_train = datasets.MNIST(
    root='./data', 
    train=True,
    download=True,
    transform=transform
)

mnist_test = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Print some basic information
print(f"Training set size: {len(mnist_train)}")
print(f"Test set size: {len(mnist_test)}")

# Let's look at a sample image
sample_img, sample_label = mnist_train[0]
print(f"Sample image shape: {sample_img.shape}")
print(f"Sample label: {sample_label}")

class FourDigitMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        
    def __len__(self):
        return len(self.mnist_dataset) // 4  # We need 4 digits per sample
        
    def __getitem__(self, idx):
        # Get 4 random digits from the dataset
        indices = random.sample(range(len(self.mnist_dataset)), 4)
        digits = [self.mnist_dataset[i][0] for i in indices]  # Get images
        labels = torch.tensor([self.mnist_dataset[i][1] for i in indices])  # Get labels as tensor
        
        # Create composite image
        composite_image = torch.zeros(1, 56, 56)
        composite_image[:, :28, :28] = digits[0]
        composite_image[:, :28, 28:] = digits[1]
        composite_image[:, 28:, :28] = digits[2]
        composite_image[:, 28:, 28:] = digits[3]
        
        return composite_image, labels

# Create the four-digit dataset
four_digit_train = FourDigitMNIST(mnist_train)
four_digit_test = FourDigitMNIST(mnist_test)

# Create dataloaders
train_loader = DataLoader(four_digit_train, batch_size=32, shuffle=True)
test_loader = DataLoader(four_digit_test, batch_size=32, shuffle=False)

# Let's look at a sample composite image
sample_img, sample_labels = four_digit_train[0]
print(f"Composite image shape: {sample_img.shape}")
print(f"Labels for the four digits: {sample_labels}")