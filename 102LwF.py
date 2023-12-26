import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Define main hyperparameters
batch_size = 64
learning_rate = 0.01
num_epochs = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a simple MLP model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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


class LwF:
    def __init__(self, model, alpha=1, temperature=2):
        self.model = model
        self.alpha = alpha
        self.temperature = temperature

    def update_model(self, new_model):
        self.model = new_model

    def train(self, old_dataloader, new_dataloader, optimizer, num_epochs):
        for epoch in range(num_epochs):
            for old_data, new_data in zip(old_dataloader, new_dataloader):
                old_inputs, old_labels, old_indexes = old_data
                new_inputs, new_labels, new_indexes = new_data

                old_inputs = old_inputs.to(device)
                old_labels = old_labels.to(device)
                new_inputs = new_inputs.to(device)
                new_labels = new_labels.to(device)

                index_list_old.append(old_indexes)
                index_list_new.append(new_indexes)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Get old task outputs with the old model
                with torch.no_grad():
                    old_outputs = self.model(old_inputs)

                # Forward pass on the new model
                new_outputs_old_task = self.model(old_inputs)
                new_outputs_new_task = self.model(new_inputs)

                # Compute old task loss (distillation loss)
                old_loss = F.kl_div(F.log_softmax(new_outputs_old_task/self.temperature, dim=1),
                                    F.softmax(old_outputs/self.temperature, dim=1),
                                    reduction='batchmean') * (self.temperature**2)

                # Compute new task loss
                new_loss = F.cross_entropy(new_outputs_new_task, new_labels)

                # Combine the two losses
                loss = self.alpha*old_loss + new_loss

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
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

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Download and load the training data
trainset = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
# Wrap the MNIST dataset in the custom dataset
indexed_dataset = IndexedMNISTDataset(trainset)
trainloader = DataLoader(indexed_dataset, batch_size=batch_size, shuffle=True)

# Download and load the test data
testset = datasets.MNIST(root='./data', download=True, train=False, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Create an instance of the LwF class
lwf = LwF(model)

index_list_new = []
index_list_old = []

# Train on the first task
lwf.train(trainloader, trainloader, optimizer, num_epochs)

print(index_list_new)
print(index_list_old)

# # Let's simulate a new task by reusing the same MNIST dataset
# # In practice, you would use a different dataset for each new task
# lwf.update_model(Net().to(device))
# lwf.train(trainloader, trainloader, optimizer, num_epochs)

lwf.test(device, testloader)