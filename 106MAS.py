import torch
from torch.nn.functional import cross_entropy
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, Dataset, DataLoader
import numpy as np

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

# Model and optimizer
model = Model()  # your PyTorch model
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Previous tasks data
prev_tasks_data = []
prev_tasks_importance = []

# Prepare the MNIST dataset
class IndexedMNISTDataset(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __getitem__(self, index):
        image, label = self.mnist_dataset[index]
        return image, label, index

    def __len__(self):
        return len(self.mnist_dataset)

# Data Preparation: Prepare two tasks from MNIST
mnist_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=mnist_transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=mnist_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Wrap the MNIST dataset in the custom dataset
indexed_dataset = IndexedMNISTDataset(train_dataset)
new_task_data = indexed_dataset

n_epochs = 2
index_list = []
mas_lambda = 0.2
# Train on new task
for epoch in range(n_epochs):
    for X, y, index in new_task_data:
        index_list.append(index)
        optimizer.zero_grad()
        output = model(X)
        loss = cross_entropy(output, y)

        # Add MAS loss for each previous task
        for task_data, task_importance in zip(prev_tasks_data, prev_tasks_importance):
            mas_loss = 0
            for p, imp in zip(model.parameters(), task_importance):
                mas_loss += (imp * (p - task_data)**2).sum()
            loss += mas_lambda * mas_loss

        loss.backward()
        optimizer.step()

    # Calculate importance for this task
    task_importance = []
    for X, y, index in new_task_data:
        model.zero_grad()
        output = model(X)
        y_grad = torch.autograd.grad(cross_entropy(output, y), model.parameters())
        task_importance.append([p.data for p in y_grad])
    prev_tasks_importance.append(task_importance)
    prev_tasks_data.append([p.data for p in model.parameters()])  # store parameter values


print(index_list)

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
    
test(model, test_loader)