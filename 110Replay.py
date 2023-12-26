import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare the MNIST dataset
class IndexedMNISTDataset(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __getitem__(self, index):
        image, label = self.mnist_dataset[index]
        return image, label, index

    def __len__(self):
        return len(self.mnist_dataset)


# Define a simple feed-forward model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fla = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(self.fla(x)))
        x = self.fc2(x)
        return x

def get_data_loaders():
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Wrap the MNIST dataset in the custom dataset
    indexed_dataset = IndexedMNISTDataset(trainset)
    train_loader = DataLoader(indexed_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False)
    return train_loader, test_loader

def train(model, device, train_loader, optimizer, index_list, replay_buffer=None, replay_ratio=0.5):
    model.train()
    for batch_idx, (data, target, index) in enumerate(train_loader):
        index_list.append(index)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # Save some examples for replay
        if replay_buffer is not None:
            replay_buffer.add(data, target)

        # Replay old examples
        if replay_buffer is not None and len(replay_buffer) > 0 and np.random.rand() < replay_ratio:
            old_data, old_target = replay_buffer.sample()
            old_data, old_target = old_data.to(device), old_target.to(device)
            optimizer.zero_grad()
            old_output = model(old_data)
            replay_loss = F.nll_loss(old_output, old_target)
            replay_loss.backward()
            optimizer.step()

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, data, target):
        if len(self.buffer) < self.capacity:
            self.buffer.append((data, target))
        else:
            self.buffer[self.position] = (data, target)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        return self.buffer[np.random.randint(len(self.buffer))]

    def __len__(self):
        return len(self.buffer)

def main():
    index_list = []
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    train_loader, test_loader = get_data_loaders()
    replay_buffer = ReplayBuffer(capacity=10000)

    for epoch in range(1, 11):
        train(model, device, train_loader, optimizer, index_list, replay_buffer)
        # test(model, device, test_loader)

    print(index_list)

    test(model, device, test_loader)

if __name__ == '__main__':
    main()
