import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch import nn, optim

# Define a simple MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.layers(x)

# Prepare the MNIST dataset
class IndexedMNISTDataset(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __getitem__(self, index):
        image, label = self.mnist_dataset[index]
        return image, label, index

    def __len__(self):
        return len(self.mnist_dataset)

# Prepare the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Wrap the MNIST dataset in the custom dataset
indexed_dataset = IndexedMNISTDataset(mnist_dataset)

# Create a data loader
train_loader = torch.utils.data.DataLoader(indexed_dataset, batch_size=64, shuffle=True)


# Initialize the model, loss function, and optimizer
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
index_list = []

# Define a function for training the model on a task
def train(model, loader, criterion, optimizer):
    model.train()
    for images, labels, indexes in loader:
        index_list.append(indexes)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

# Train the model on the first task
train(model, train_loader, criterion, optimizer)
print(index_list)
# Prepare the MNIST test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

def test(model, loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    # Don't calculate gradients for efficiency
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy: {}%'.format(100 * correct / total))

# Test the model
test(model, test_loader)

