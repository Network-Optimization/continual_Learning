import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, Dataset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple Feed Forward model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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

indices = np.arange(len(train_dataset))
labels = train_dataset.train_labels.numpy()

# create two tasks: one for digits 0-4 and another for digits 5-9
task1_indices = indices[labels < 5]
task2_indices = indices[labels >= 5]

task1_dataset = Subset(indexed_dataset, task1_indices)
task2_dataset = Subset(indexed_dataset, task2_indices)

# Prepare data loaders
task1_loader = torch.utils.data.DataLoader(task1_dataset, batch_size=64, shuffle=True)
task2_loader = torch.utils.data.DataLoader(task2_dataset, batch_size=64, shuffle=True)
test_loader =  torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# Initialize the model
model = Model().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Define the EWC class
class EWC(object):
    def __init__(self, model: nn.Module, dataloader: torch.utils.data.DataLoader):
        self.model = model
        self.dataloader = dataloader
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad} 
        self._means = {}  # Parameter means
        self._precision_matrices = self._calculate_importance()  # Fisher information

        for n, p in self.params.items():
            self._means[n] = p.clone().detach()

    def _calculate_importance(self):
        print('Computing Fisher information..')
        precision_matrices = {}
        for n, p in self.params.items():
            p.data.zero_()
            precision_matrices[n] = p.clone().detach()

        self.model.eval()
        for input, target, index in self.dataloader:
            self.model.zero_grad()
            input = input.to(device)
            target = target.to(device)
            output = F.log_softmax(self.model(input), dim=1)
            loss = F.nll_loss(output, target)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.dataloader)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss
    
    def test(self, device, test_loader):
        self.model.eval()  # Set the model to evaluation mode
        test_loss = 0
        correct = 0
        with torch.no_grad():  # No gradients are required since we're not updating the model parameters
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                # Sum up batch loss
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                # Get the index of the max log-probability (the predicted class label)
                pred = output.argmax(dim=1, keepdim=True) 
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

index_list1 = []
index_list2 = []
# Training the model on task 1
model.train()
for epoch in range(10):
    for batch_idx, (data, target, index) in enumerate(task1_loader):
        index_list1.append(index)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

# EWC on task 1
ewc = EWC(model, task1_loader)

print(index_list1)

ewc.test(device, test_loader)