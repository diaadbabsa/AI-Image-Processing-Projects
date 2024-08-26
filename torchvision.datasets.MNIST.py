import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Simple deep neural network with Sigmoid activation (to demonstrate vanishing gradient)
class SimpleNetSigmoid(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNetSigmoid, self).__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

# Simple deep neural network with ReLU activation (to mitigate vanishing gradient)
class SimpleNetReLU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNetReLU, self).__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

# Function to compute accuracy
def compute_accuracy(net, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = net(images.view(-1, 28*28))
            predicted = (outputs > 0).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Function to train the network and plot the accuracy
def train_and_plot(net_sigmoid, net_relu, criterion, optimizer_sigmoid, optimizer_relu, train_loader, num_epochs=10):
    accuracies_sigmoid = []
    accuracies_relu = []

    for epoch in range(num_epochs):
        # Train Sigmoid Network
        for images, labels in train_loader:
            labels = labels.float().view(-1, 1)  # Reshape labels for binary classification
            
            optimizer_sigmoid.zero_grad()
            outputs_sigmoid = net_sigmoid(images.view(-1, 28*28))
            loss_sigmoid = criterion(outputs_sigmoid, labels)
            loss_sigmoid.backward()
            optimizer_sigmoid.step()
        
        accuracy_sigmoid = compute_accuracy(net_sigmoid, train_loader)
        accuracies_sigmoid.append(accuracy_sigmoid)

        # Train ReLU Network
        for images, labels in train_loader:
            labels = labels.float().view(-1, 1)  # Reshape labels for binary classification
            
            optimizer_relu.zero_grad()
            outputs_relu = net_relu(images.view(-1, 28*28))
            loss_relu = criterion(outputs_relu, labels)
            loss_relu.backward()
            optimizer_relu.step()

        accuracy_relu = compute_accuracy(net_relu, train_loader)
        accuracies_relu.append(accuracy_relu)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Sigmoid plot
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), accuracies_sigmoid, label="Sigmoid")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs. Epochs: {net_sigmoid.__class__.__name__}')
    
    # ReLU plot
    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), accuracies_relu, label="ReLU", color="green")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs. Epochs: {net_relu.__class__.__name__}')

    plt.tight_layout()
    plt.show()

# Hyperparameters
input_size = 28*28
hidden_size = 128
output_size = 1
learning_rate = 0.01
batch_size = 64
num_epochs = 10

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize networks
net_sigmoid = SimpleNetSigmoid(input_size, hidden_size, output_size)
net_relu = SimpleNetReLU(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer_sigmoid = optim.SGD(net_sigmoid.parameters(), lr=learning_rate)
optimizer_relu = optim.SGD(net_relu.parameters(), lr=learning_rate)

# Train and plot for both networks
train_and_plot(net_sigmoid, net_relu, criterion, optimizer_sigmoid, optimizer_relu, train_loader, num_epochs)
