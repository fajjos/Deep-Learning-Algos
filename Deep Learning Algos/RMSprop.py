import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Load MNIST dataset
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser='auto')
X = mnist.data.astype('float32')
y = mnist.target.astype('int')
X /= 255.0  # Normalize input

# Split the dataset into training and testing sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training set to use only a portion (10% or 25%)
X_train, _, y_train, _ = train_test_split(X_train_full, y_train_full, test_size=0.9, random_state=42)  # Change test_size to 0.9 for 10% or 0.75 for 25%

# Convert labels to one-hot encoding
label_binarizer = LabelBinarizer()
y_train_one_hot = label_binarizer.fit_transform(y_train)
y_test_one_hot = label_binarizer.transform(y_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train_one_hot, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test_one_hot, dtype=torch.float32)

# Define a simple neural network model using PyTorch
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

# Train the model
def train_model_rmsprop(model, criterion, optimizer, num_epochs=10):
    train_losses = []
    test_losses = []
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_train_loss = 0.0
        epoch_test_loss = 0.0
        model.train()
        for i in range(len(X_train_tensor)):
            optimizer.zero_grad()
            output = model(X_train_tensor[i].unsqueeze(0))
            loss = criterion(output, y_train_tensor[i].unsqueeze(0))
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss / len(X_train_tensor))

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            loss = criterion(outputs, y_test_tensor)
            epoch_test_loss = loss.item()
        test_losses.append(epoch_test_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]}, Test Loss: {test_losses[-1]}")

    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training Time: {training_time} seconds")
    return train_losses, test_losses

# Initialize the model, loss function, and optimizer with RMSprop optimizer
model_rmsprop = NeuralNetwork()
criterion_rmsprop = nn.BCELoss()
optimizer_rmsprop = optim.RMSprop(model_rmsprop.parameters(), lr=0.001)

# Train the model with RMSprop optimizer
train_losses_rmsprop, test_losses_rmsprop = train_model_rmsprop(model_rmsprop, criterion_rmsprop, optimizer_rmsprop, num_epochs=10)

# Plot the loss curves
plt.plot(train_losses_rmsprop, label='Training Loss')
plt.plot(test_losses_rmsprop, label='Test Loss')
plt.title('Training and Test Loss (RMSprop Optimizer)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
