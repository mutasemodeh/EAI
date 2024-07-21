import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset,random_split
from tqdm import tqdm
from sklearn.cluster import KMeans
import torch.optim as optim


TRAIN_VAL_DATASET_RATIO = 0.8
IMG_WIDTH = 50
NUM_CLUSTER=400
def extract_edge_coordinates(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply edge detection (Canny edge detector)
    edges = cv2.Canny(image, 1, 200)
    
    # Extract edge coordinates (x, y)
    coordinates = np.column_stack(np.where(edges > 0))
    
    return coordinates

def plot_edge_coordinates(coordinates):
    plt.figure(figsize=(5, 5))
    plt.scatter(coordinates[:, 1], coordinates[:, 0], s=1, color='red')
    plt.gca().invert_yaxis()
    plt.title('Edge Coordinates')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def extract_hu_moments(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_WIDTH, IMG_WIDTH))
    edges = cv2.Canny(image, 100, 200)
    moments = cv2.moments(edges)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments

def normalize_coordinates(coordinates, img_shape):
    height, width = img_shape
    normalized_coords = coordinates / np.array([height, width])
    return normalized_coords

def cluster_coordinates(edge_coordinates, img_shape,num_clusters):
    height, width = img_shape
    normalized_coords = normalize_coordinates(edge_coordinates, img_shape)
    
    num_coords = normalized_coords.shape[0]
    
    if num_coords < num_clusters:
        # Duplicate coordinates to ensure we have enough data points
        indices = np.random.choice(num_coords, num_clusters, replace=True)
        normalized_coords = normalized_coords[indices]
    
    # Perform clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(normalized_coords)
    cluster_centers = kmeans.cluster_centers_
    
    return cluster_centers

class EdgeDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
def prepare_data_loaders(dataset_path,num_clusters,train_val_ratio=0.8, batch_size=32):
    data = []
    labels = []
    
    # Load classes and their paths
    classes = os.listdir(dataset_path)
    
    for label, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                
                # Extract features
                edge_coordinates = extract_edge_coordinates(image_path)
                if len(edge_coordinates) != 0:
                    coordinates_vector = cluster_coordinates(edge_coordinates, (IMG_WIDTH, IMG_WIDTH),num_clusters)  
                    data.append(coordinates_vector)
                    labels.append(label)
                else:
                    print(f'No edges detected in {image_path}')
    
    # Convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    
    # Check if all feature vectors have the same length
    data_lengths = [len(f) for f in data]
    if len(set(data_lengths)) != 1:
        raise ValueError(f"Inconsistent feature vector lengths: {set(data_lengths)}")

    # Convert numpy arrays to PyTorch tensors
    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Create dataset
    dataset = EdgeDataset(data_tensor, labels_tensor)
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(train_val_ratio * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, len(classes)

def train_model(train_loader, val_loader, num_classes, num_points, epochs=10, lr=0.001):
    model = PointNet(num_classes=num_classes, num_points=num_points)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_data, batch_labels in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_data.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = model(batch_data)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')

    print('Training complete!')

class TNet(nn.Module):
    def __init__(self, num_points):
        super(TNet, self).__init__()
        self.num_points = num_points

        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)  # 3x3 matrix = 9 elements

    def forward(self, x):
        batch_size = x.size(0)
        x = self.mlp1(x)
        x = F.max_pool1d(x, self.num_points)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        transform = self.fc3(x)
        transform = transform.view(-1, 3, 3)
        return transform

class PointNet(nn.Module):
    def __init__(self, num_classes, num_points):
        super(PointNet, self).__init__()
        self.num_points = num_points
        self.num_classes = num_classes

        self.tnet = TNet(num_points)
        
        # MLP layers for feature extraction
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1),
            nn.ReLU()
        )

        self.global_feat = nn.Sequential(
            nn.MaxPool1d(kernel_size=num_points),
            nn.Flatten()
        )

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.transpose(1, 2)  # Transpose to (B, C, N) format
        transform = self.tnet(x)
        x = torch.bmm(x, transform)  # Apply the learned transformation
        x = self.mlp1(x)
        x = self.global_feat(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


f='/Users/modeh/EAI2/Test2_Dataset/JPEG/Wing_Nut/90866A030_Zinc-Plated Steel Wing Nut_sim_1_xy_var_1.jpg'
x=cluster_coordinates(extract_edge_coordinates(f),(IMG_WIDTH,IMG_WIDTH),NUM_CLUSTER)
plot_edge_coordinates(x)
 

# dataset_path = '/Users/modeh/EAI2/Test_Dataset/JPEG'
# train_loader, val_loader, num_classes = prepare_data_loaders(dataset_path,NUM_CLUSTER)
# train_model(train_loader, val_loader, num_classes, NUM_CLUSTER, epochs=10)
