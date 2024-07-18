import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
TRAIN_VAL_DATASET_RATIO=0.8
IMG_WIDTH=56


from torch.utils.tensorboard import SummaryWriter

# Define transform for preprocessing images
transform = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_WIDTH)),  # Resize to desired input size
    transforms.Grayscale(),         # Convert to grayscale
    transforms.ToTensor(),          # Convert PIL image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1]
])

# Load dataset using torchvision.datasets.ImageFolder
Training_Data = '/Users/modeh/EAI2/Type_Dataset/JPEG'
dataset = datasets.ImageFolder(Training_Data, transform=transform)
# Get the class names
print("Number of classes:", len(dataset.classes))
# Split dataset into train and val
train_size = int(TRAIN_VAL_DATASET_RATIO * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create DataLoader for training and validation sets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define model architecture
class DeepEdgeNet(nn.Module):
    def __init__(self):
        super(DeepEdgeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # Reduced output channels
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # Reduced output channels
        self.fc1 = nn.Linear(16 * IMG_WIDTH * IMG_WIDTH, 64)  # Reduced input size and number of neurons
        self.fc2 = nn.Linear(64, len(dataset.classes))  # Output size for 3 classes


    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = DeepEdgeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
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

# Train and evaluate
train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=10)
