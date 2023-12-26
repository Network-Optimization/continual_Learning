import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
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

def train(model, dataloader, episodic_memory, criterion, optimizer, index_list):
    for images, labels, indexes in dataloader:
        index_list += indexes
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        # Compute gradients on the current task
        loss.backward(retain_graph=True)

        # Store gradients of the current task
        current_gradients = []
        for param in model.parameters():
            current_gradients.append(param.grad.clone())
        
        optimizer.zero_grad()

        # Compute gradients on the episodic memory
        for mem_images, mem_labels, mem_indexes in episodic_memory:
            mem_outputs = model(mem_images)
            mem_loss = criterion(mem_outputs, mem_labels)
            mem_loss.backward(retain_graph=True)

        # Average gradients
        for idx, param in enumerate(model.parameters()):
            param.grad += current_gradients[idx]
            param.grad /= 2

        optimizer.step()

def test(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad():  # No gradients are required since we're not updating the model parameters
        for data, target in test_loader:
            # data, target = data.to(device), target.to(device)
            output = model(data)
            # Sum up batch loss
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            # Get the index of the max log-probability (the predicted class label)
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# Data
train_set = MNIST(root='./data', train=True, transform=ToTensor(), download=True)
test_set = MNIST(root='./data', train=False, transform=ToTensor(), download=True)
# Wrap the MNIST dataset in the custom dataset
indexed_dataset = IndexedMNISTDataset(train_set)
train_loader = DataLoader(indexed_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

# Model, criterion, optimizer
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01)

# Episodic memory
episodic_memory = []

index_list = []

# Training
for epoch in range(2):
    train(model, train_loader, episodic_memory, criterion, optimizer, index_list)

    # Update episodic memory with some examples from the current task
    episodic_memory.append(next(iter(train_loader)))

print(index_list)

test(model, test_loader)