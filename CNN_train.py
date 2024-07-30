import os
import typer
import sys
import signal
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models

app = typer.Typer()
CNN_FACTOR=8

TRAIN_VAL_DATASET_RATIO = 0.8
IMG_WIDTH = 250

transform = transforms.Compose(
    [
        transforms.Resize((IMG_WIDTH, IMG_WIDTH)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

class DeepEdgeNet(nn.Module):
    def __init__(self, num_classes):
        super(DeepEdgeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1=nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2=nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.bn3=nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(32 *IMG_WIDTH *IMG_WIDTH, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
       
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x=self.conv1(x)
        # x=self.bn1(x)
        x=torch.relu(x)
        x=self.conv2(x)
        # x=self.bn2(x)
        x=torch.relu(x)
        # x=self.conv3(x)
        # # x=self.bn2(x)
        # x=torch.relu(x)
        # x = self.adaptive_pool(x)
        
        x=self.flatten(x)
        x=self.fc1(x)
        x=torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
class VGG16Custom(nn.Module):
    def __init__(self, num_classes):
        super(VGG16Custom, self).__init__()
        # Load the pretrained VGG16 model
        self.vgg16 = models.vgg16(pretrained=True)
        
        # Modify the first convolutional layer to accept grayscale images
        self.vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        
        # Freeze the layers except the final classifier layers
        for param in self.vgg16.parameters():
            param.requires_grad = False
        
        # Modify the classifier to match the number of classes
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        x = self.vgg16(x)
        return x

def prepare_data_loaders(training_data_path):
    dataset = datasets.ImageFolder(training_data_path, transform=transform)
    train_size = int(TRAIN_VAL_DATASET_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader, dataset.classes


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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"CNN_model_{accuracy:.2f}_{timestamp}.pth"
    model_path = os.path.join(model_dir, filename)

    # Save the model to the specified path
    torch.save(model.state_dict(), model_path)

    # Print accuracy with the specific tag
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Model saved to: {model_path}")


def train_deep_edge_net(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    num_epochs=10,
    early_stopping_patience=3,
    model_dir="models",
):
    # Scheduler to reduce learning rate when a metric has stopped improving
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3, verbose=True
    )

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
            for inputs, labels in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"
            ):
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

            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Training Loss: {epoch_loss:.4f}, "
                f"Validation Loss: {val_loss:.4f}, "
                f"Validation Accuracy: {val_accuracy:.2f}%"
            )

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


@app.command()
def main(
    base_dir: str = typer.Option("/Users/modeh/EAI2"),
    dataset: str = typer.Option("Length_Dataset"),
    num_epochs: int = typer.Option(10),
    early_stopping_patience: int = typer.Option(3),
):
    base_dir = Path(base_dir) / dataset
    training_data_path = str(base_dir / "JPEG")
    model_dir = str(base_dir / "Model")
    train_loader, val_loader, classes = prepare_data_loaders(training_data_path)

    # # # Prepare the data DeepEdgeNet model
    model = DeepEdgeNet(len(classes))
    model = VGG16Custom(len(classes))
    criterion = nn.CrossEntropyLoss()  # noqa: F841
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Train and evaluate DeepEdgeNet
    train_deep_edge_net(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        model_dir=model_dir,
    )

    sys.exit(0)


if __name__ == "__main__":
    app()
