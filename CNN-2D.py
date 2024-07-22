import os
import time
import cv2
import sys
import signal
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


TRAIN_VAL_DATASET_RATIO = 0.8
IMG_WIDTH = 250


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

transform = transforms.Compose([
transforms.Resize((IMG_WIDTH, IMG_WIDTH)),
transforms.Grayscale(),
transforms.ToTensor(),
transforms.Normalize((0.5,), (0.5,))
])

def prepare_data_loaders(training_data_path):

    dataset = datasets.ImageFolder(training_data_path, transform=transform)
    train_size = int(TRAIN_VAL_DATASET_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader, len(dataset.classes)

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

def load_model(model, model_path):
    """
    Load the model's state dictionary from a specific file path.

    Parameters:
    - model: The model architecture to load the weights into
    - model_path: Path to the saved model file

    Returns:
    - model: The model with weights loaded
    """
    # Load the state dictionary from the file
    model.load_state_dict(torch.load(model_path))
    
    # Set the model to evaluation mode
    model.eval()
    
    print(f'Model loaded from: {model_path}')
    return model

def train_deep_edge_net(model, criterion, optimizer, train_loader, val_loader, num_epochs=10, early_stopping_patience=5, model_dir='models'):
    # Scheduler to reduce learning rate when a metric has stopped improving
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    best_val_accuracy = 0
    best_model_state = None
    epochs_no_improve = 0


    def signal_handler(sig, frame):
        print("Signal received, saving the best model...")
        if best_model_state:
            model.load_state_dict(best_model_state)
            save_model(model, best_val_accuracy, model_dir)
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
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
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_accuracy = (correct / total) * 100
            val_loss /= len(val_loader)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Training Loss: {epoch_loss:.4f}, '
                  f'Validation Loss: {val_loss:.4f}, '
                  f'Validation Accuracy: {val_accuracy:.2f}%')
            
            scheduler.step(val_loss)
            
            # Track the best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model_state = model.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping")
                break

    finally:
        # Save the best model at the end
        if best_model_state:
            model.load_state_dict(best_model_state)
            save_model(model, best_val_accuracy, model_dir)

    return best_val_accuracy

def calculate_class_accuracies(model, val_loader):
    """
    Calculate the accuracy for each class and print the class names with their accuracies.

    Parameters:
    - model: The trained model
    - val_loader: DataLoader for the validation set

    Returns:
    - class_accuracies: A dictionary where keys are class indices and values are accuracies
    """
    model.eval()
    # Access the original dataset
    original_dataset = val_loader.dataset.dataset
    class_names = original_dataset.classes  # Extract class names from the original dataset
    num_classes = len(class_names)
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(labels)):
                label = labels[i].item()
                if predicted[i] == label:
                    class_correct[label] += 1
                class_total[label] += 1
    
    class_accuracies = {i: class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)}
    
    # Print class names and their accuracies
    for i, class_name in enumerate(class_names):
        accuracy = class_accuracies[i] * 100  # Convert to percentage
        print(f'Class: {class_name}, Accuracy: {accuracy:.2f}%')
    
    return class_accuracies


def main():

    base_dir = Path('/Users/modeh/EAI2/Type_Dataset')
    training_data_path = str(base_dir / 'JPEG')
    model_dir=str(base_dir / 'Model')

    # # Prepare the data DeepEdgeNet model
    train_loader, val_loader, num_classes = prepare_data_loaders(training_data_path)
    model = DeepEdgeNet(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # Train and evaluate DeepEdgeNet
    train_deep_edge_net(model, criterion, optimizer, 
                        train_loader, val_loader, 
                        num_epochs=10, early_stopping_patience=5, model_dir=model_dir)

    # model_path='/Users/modeh/EAI2/Test2_Dataset/Model/CNN_model_74.70_20240721_185933.pth'
    # load_model(model, model_path)
    # calculate_class_accuracies(model, train_loader)

if __name__ == "__main__":
    main()
