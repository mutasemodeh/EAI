import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import time
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image
from datetime import datetime

TRAIN_VAL_DATASET_RATIO = 0.8
IMG_WIDTH = 200

# Define transform for preprocessing images
transform = transforms.Compose([
transforms.Resize((IMG_WIDTH, IMG_WIDTH)),
transforms.Grayscale(),
transforms.ToTensor(),
transforms.Normalize((0.5,), (0.5,))
])

# Function to prepare data loaders
def prepare_data_loaders(training_data_path):

    dataset = datasets.ImageFolder(training_data_path, transform=transform)
    train_size = int(TRAIN_VAL_DATASET_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader, len(dataset.classes)

class DeepEdgeNet(nn.Module):
    def __init__(self, num_classes):
        super(DeepEdgeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * IMG_WIDTH * IMG_WIDTH, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Train and evaluate the DeepEdgeNet model
def train_deep_edge_net(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_accuracy = (correct / total) * 100
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Training Loss: {epoch_loss:.4f}, '
              f'Validation Accuracy: {val_accuracy:.2f}%')
        
    return val_accuracy

def save_model(model, accuracy, model_dir):
    """
    Save the trained model to a specific directory with a filename including accuracy and timestamp,
    and print the accuracy with a specific tag.

    Parameters:
    - model: The trained model
    - accuracy: Accuracy achieved during validation
    - model_dir: Directory to save the trained model
    - tag: Tag to print the accuracy with
    """
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Generate filename with accuracy and timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'CNN_model_{accuracy:.2f}_{timestamp}.pth'
    model_path = os.path.join(model_dir, filename)
    
    # Save the model to the specified path
    torch.save(model.state_dict(), model_path)
    
    # Print accuracy with the specific tag
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Model saved to: {model_path}')

def main():

    base_dir = Path('/Users/modeh/EAI2/Type_Dataset')
    training_data_path = str(base_dir / 'JPEG')
    model_dir=str(base_dir / 'Model')

    # Initialize DeepEdgeNet model
    train_loader, val_loader, num_classes = prepare_data_loaders(training_data_path)
    model = DeepEdgeNet(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train and evaluate DeepEdgeNet
    accuracy=train_deep_edge_net(model, criterion, optimizer, train_loader, val_loader, num_epochs=10)
    save_model(model, accuracy, model_dir)


if __name__ == "__main__":
    main()
