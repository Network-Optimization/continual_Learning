import torch
import torch.nn as nn
import torch.optim as optim
import torch
from torch.nn.functional import cross_entropy, mse_loss
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, Dataset, DataLoader
import numpy as np

class FeatureExtractor(nn.Module):
    # You need to define your feature extractor model (like a CNN) here
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # ...add more layers as necessary
            )
    
    def forward(self, input):
        return self.features(input)

class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(Classifier, self).__init__()
        self.layer = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        return self.layer(x)


# Training setup
feature_extractor = FeatureExtractor()
classifier_head = Classifier(feature_dim=14, num_classes=10)

# Previous tasks data
old_task_loader = []

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

index_list = []

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST

# Data Preparation: Prepare two tasks from MNIST
mnist_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))])

def create_task_dataset():
    # Load MNIST
    mnist_train = MNIST('./data', train=True, download=True, transform=mnist_transform)
    mnist_test = MNIST('./data', train=False, download=True, transform=mnist_transform)
    # Wrap the MNIST dataset in the custom dataset
    indexed_dataset = IndexedMNISTDataset(mnist_train)
    task_loader = DataLoader(indexed_dataset, batch_size=64, shuffle=True)
    
    # Define the task boundaries
    task_classes = [[0,1], [2,3], [4,5], [6,7], [8,9]]
    
    task_datasets_train = []
    task_datasets_test = []

    for task in task_classes:
        task_indices_train = [(target in task) for target in indexed_dataset.mnist_dataset.targets]
        task_indices_test = [(target in task) for target in mnist_test.targets]

        task_dataset_train = torch.utils.data.dataset.Subset(indexed_dataset, [i for i, x in enumerate(task_indices_train) if x])
        task_dataset_test = torch.utils.data.dataset.Subset(mnist_test, [i for i, x in enumerate(task_indices_test) if x])
        
        task_datasets_train.append(task_dataset_train)
        task_datasets_test.append(task_dataset_test)
    
    return task_datasets_train, task_datasets_test

# Create data loaders for each task
task_datasets_train, task_datasets_test = create_task_dataset()
data_loaders_train = [DataLoader(dataset, batch_size=64, shuffle=True) for dataset in task_datasets_train]
data_loaders_test = [DataLoader(dataset, batch_size=64, shuffle=False) for dataset in task_datasets_test]

num_epochs = 2

# For training, you can now loop over the tasks and use the corresponding data loader for training. For the old data, you can concatenate the data loaders of the previous tasks.
for task, data_loader in enumerate(data_loaders_train):
    if task > 0:
        # Combine old data loaders
        old_data_loaders = data_loaders_train[:task]
        old_data_loader_combined = DataLoader(ConcatDataset([dl.dataset for dl in old_data_loaders]), batch_size=64, shuffle=True)
        # Now you have a combined data loader for the old data and a separate one for the new task
    else:
        old_data_loader_combined = data_loaders_train[:task]
    # Continue with training...
    # You should use an appropriate loss function here
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(feature_extractor.parameters()) + list(classifier_head.parameters()), lr=0.01)

    # Training loop
    for epoch in range(num_epochs):
        for data, labels, index in old_data_loader_combined:
            index_list.append(index)
            # optimizer.zero_grad()
            # features = feature_extractor(data)
            # output = classifier_head(features)
            # loss = mse_loss(output, labels)
            # loss.backward()
            # optimizer.step()

    # After training, perform bias correction on classifier head

# Evaluation...
print(index_list)

