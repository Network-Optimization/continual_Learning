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
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# Initialize the model
model = Model().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# SI
class SI(object):
    def __init__(self, model):
        self.model = model
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._previous_params = {n: p.clone().detach() for n, p in self.params.items()}
        self._importance = {n: torch.zeros_like(p) for n, p in self.params.items()}
        self._omega = {n: torch.zeros_like(p) for n, p in self.params.items()}

    def update(self):
        for n, p in self.params.items():
            delta_theta = p.detach() - self._previous_params[n]
            self._omega[n] += self._importance[n] * delta_theta ** 2
            self._previous_params[n] = p.clone().detach()

    def penalty(self):
        penalty = 0
        for n, p in self.model.named_parameters():
            penalty += (self._omega[n] * (p - self._previous_params[n]) ** 2).sum()
        return penalty

    def after_backward(self):
        for n, p in self.params.items():
            self._importance[n] = -p.grad.detach() * (p.detach() - self._previous_params[n])
    
    

index_list1 = []
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

print(index_list1)

model.test(device, test_loader)


