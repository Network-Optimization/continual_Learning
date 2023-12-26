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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PODNet(nn.Module):
    def __init__(self, num_classes):
        super(PODNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # ...add more layers as necessary
        )
        self.classifier = nn.Linear(12544, num_classes)
        self.distilled_outputs = []  # Store the distilled outputs for each task

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # print(x.size())
        return self.classifier(x)


import torch
from torch import optim
from torchvision import datasets, transforms


# model definition
model = PODNet(num_classes=10)

# setup optimizer
optimizer = optim.Adam(model.parameters(), lr=0.003)

# define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model = model.to(device)

# Previous tasks data
old_task_loaders = []

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


# Wrap the MNIST dataset in the custom dataset
indexed_dataset = IndexedMNISTDataset(train_dataset)
new_task_loader = DataLoader(indexed_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

T = 10
alpha = 0.1
index_list = []

# training loop
for epoch in range(2):
    for (inputs, labels, indexes) in new_task_loader:
        index_list.append(indexes)
        # inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.to(device)
        optimizer.zero_grad()

        # Compute the standard classification loss
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)

        # Compute the distillation loss for each old task
        for task_id, old_task_loader, index in enumerate(old_task_loaders):
            old_inputs, _ = next(iter(old_task_loader))
            old_inputs = old_inputs.to(device)
            old_outputs = model(old_inputs)
            distill_outputs = model.distilled_outputs[task_id]
            distill_loss = (F.softmax(old_outputs / T, dim=1) - F.softmax(distill_outputs / T, dim=1)).pow(2).mean()

            # Add the distillation loss to the total loss, with the distillation weight
            loss += alpha * distill_loss

        loss.backward()
        optimizer.step()

    # At the end of an epoch, update the distilled outputs for the new task
    model.distilled_outputs.append(torch.cat([model(inputs.to(device)).detach() for inputs, labels, indexes in new_task_loader], dim=0))

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