import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

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
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 10)

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

def train(model, device, train_loader, optimizer, index_list):
    model.train()
    for data, target, index in train_loader:
        index_list.append(index)
        data, target = data.to(device), target.to(device)
        # print(data.size())
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

def icarl_train(model, device, train_loader, optimizer, exemplars, index_list):
    model.train()
    for data, target, index in train_loader:
        index_list.append(index)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        # Compute the classification loss
        class_loss = F.nll_loss(output, target)

        # If exemplars are available, compute the distillation loss
        if exemplars is not None:
            old_target = model(exemplars)
            distill_loss = F.kl_div(F.log_softmax(output, dim=1), F.softmax(old_target, dim=1))
            loss = class_loss + distill_loss
        else:
            loss = class_loss

        loss.backward()
        optimizer.step()

def main():
    index_list = []
    train_loader, test_loader = get_data_loaders()
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Phase 1: Learn on initial dataset
    for epoch in range(1, 2):
        train(model, device, train_loader, optimizer, index_list)

    # Save model parameters
    old_model = model.state_dict()

    # Generate exemplars (this is just a placeholder, you need to implement the actual exemplar generation process)
    exemplars = None

    # Phase 2: Learn on new tasks, using iCaRL
    for task in range(2, 11):
        # Assume new_task_loader is your new data for this task
        new_task_loader, _ = get_data_loaders()
        for epoch in range(1, 2):
            icarl_train(model, device, new_task_loader, optimizer, exemplars, index_list)

        # Save new model parameters and generate new exemplars
        old_model = model.state_dict()
        exemplars = None  # Update this with new exemplars

    print(index_list)
    def test(model, test_loader, device):
        model.eval()  # Set the model to evaluation mode
        test_loss = 0
        correct = 0
        with torch.no_grad():  # No gradients are required since we're not updating the model parameters
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
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
    
    test(model, test_loader, device)

if __name__ == '__main__':
    main()
