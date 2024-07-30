import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import transforms
from zernike import RZern
import os
from datetime import datetime
import joblib  # Import joblib for saving the models
import networkx as nx
import torch
from sklearn.cluster import KMeans

# from sklearn.neighbors import NearestNeighborsconda
# from torch_geometric.data import Data

VAL_FRACTION = 0.2
IMG_WIDTH = 2000



def transformer(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_WIDTH, IMG_WIDTH))
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.5)
    # image = image / np.max(image)  # Normalize to range [0, 1]
    return blurred_image

def detect_edges(image, debug=False):
    edges = cv2.Canny(image, 100, 200)
    edges_coordinates = np.column_stack(np.where(edges > 0))

    if debug:
        image_with_edges = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for x, y in edges_coordinates:
            cv2.circle(image_with_edges, (y, x), 5, (0, 255, 0), -1)
        plt.figure(figsize=(10, 5))
        plt.imshow(image_with_edges)
        plt.title("Edges Detected")
        plt.axis("off")
        plt.show()

    return edges_coordinates,edges


def detect_corners(image, debug=False):
    corners = cv2.goodFeaturesToTrack(
        image, maxCorners=100, qualityLevel=0.01, minDistance=10
    )
    corners = np.int0(corners)

    if debug:
        image_with_corners = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(image_with_corners, (x, y), 5, (0, 255, 0), -1)
        plt.figure(figsize=(10, 5))
        plt.imshow(image_with_corners)
        plt.title("Corners Detected")
        plt.axis("off")
        plt.show()

    return corners


def detect_contours(image, debug=False):
    edges_coordinates,edges = detect_edges(image)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    flattened_contours = np.vstack(contours)

    if debug:
        image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
        plt.figure(figsize=(10, 5))
        plt.imshow(image_with_contours)
        plt.title("Contours Detected")
        plt.axis("off")
        plt.show()

    return flattened_contours


def detect_lines(image, debug=False):
    edges_coordinates,edges = detect_edges(image)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10
    )

    if debug:
        image_with_lines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 5)
        plt.figure(figsize=(10, 5))
        plt.imshow(image_with_lines)
        plt.title("Lines Detected")
        plt.axis("off")
        plt.show()

    return lines


def detect_circles(image, debug=False):
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=200,
        param2=50,
        minRadius=10,
        maxRadius=int(IMG_WIDTH / 2),
    )

    if debug:
        image_with_circles = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]
                cv2.circle(image_with_circles, center, 1, (0, 100, 100), 5)
                cv2.circle(image_with_circles, center, radius, (255, 0, 255), 5)
        plt.figure(figsize=(10, 5))
        plt.imshow(image_with_circles)
        plt.title("Circles Detected")
        plt.axis("off")
        plt.show()

    return circles


def extract_hu_moments(image, debug=False):
    # 7 disrciportrs (invarient to rotation and scale, and translation)
    # Compute moments
    moments = cv2.moments(image)

    # Compute Hu Moments
    hu_moments = cv2.HuMoments(moments).flatten()

    # If debug flag is set, visualize the image and Hu Moments
    if debug:
        # Display the image
        plt.imshow(image, cmap="gray")
        plt.title("Image for Hu Moments")
        plt.axis("off")  # Hide axes
        plt.show()

        # Print Hu Moments
        print("Hu Moments:")
        for i, hu_moment in enumerate(hu_moments):
            print(f"Hu Moment {i+1}: {hu_moment}")

    return hu_moments


def compute_zernike_moments(image, max_order=8, debug=False):
    # Ensure the image is in the range [0, 1]
    if image.max() > 1:
        image = image / 255.0
    
    # Prepare Zernike calculator
    cart = RZern(max_order)
    L, K = image.shape
    ddx = np.linspace(-1.0, 1.0, K)
    ddy = np.linspace(-1.0, 1.0, L)
    xv, yv = np.meshgrid(ddx, ddy)
    
    # Create a circular mask to avoid NaNs
    mask = np.sqrt(xv**2 + yv**2) <= 1.0
    cart.make_cart_grid(xv, yv)

    zernike_moments = np.zeros(cart.nk)

    # Compute Zernike moments
    for i in range(cart.nk):
        Phi = cart.eval_grid(np.eye(cart.nk)[:, i], matrix=True)
        zernike_moments[i] = np.sum(image[mask] * Phi[mask])

    if debug:
        plt.figure(figsize=(10, 5))
        plt.plot(range(cart.nk), zernike_moments, marker='o')
        plt.title("Zernike Moments")
        plt.xlabel("Mode Index")
        plt.ylabel("Moment Value")
        plt.grid(True)
        plt.show()

    return zernike_moments

def extract_sift_features(image, debug=False):
    # good for intricate detailes images and texture (invarient to rotation and scale, based on blolbs matching)

    # Create a SIFT detector object
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)

    # If no descriptors are found, return a zero vector
    if descriptors is None:
        if debug:
            print("No descriptors found.")
        return np.zeros(128)

    # Limit descriptors to 128 elements
    descriptors = descriptors.flatten()[:128]

    # If debug flag is set, visualize keypoints
    if debug:
        # Draw keypoints on the image
        image_with_keypoints = cv2.drawKeypoints(
            image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        # Convert image from BGR to RGB (OpenCV uses BGR by default)
        image_with_keypoints = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)

        # Display the image with keypoints
        plt.imshow(image_with_keypoints)
        plt.title("SIFT Keypoints")
        plt.axis("off")  # Hide axes
        plt.show()

    return descriptors

    # similar to SIFT but faster

    image = transformer(image_path)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    if descriptors is None:
        return np.zeros(128)
    return descriptors.flatten()[:128]  # Limit to 128 elements


def extract_fft_features(image, debug=False):
    # good for periodic structres and their singutre
    # Compute FFT
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)

    # Compute magnitude spectrum
    magnitude_spectrum = np.abs(fft_shifted)
    # Compute radial average of the magnitude spectrum
    avg_magnitude = radial_average(magnitude_spectrum)

    if debug:
        plt.figure(figsize=(10, 5))
        plt.plot(np.log10(avg_magnitude))
        plt.title("Radial Averaged Magnitude Spectrum")
        plt.xlabel("Radius")
        plt.ylabel("Log10 Magnitude")
        plt.legend([f"Image {i+1}" for i in range(len(image))])
        plt.show()

    return avg_magnitude 


def combine_features(image_path):
    # combine multipe of more features such that they can be classified normally

    # Extract Hu Moments
    hu_moments = extract_hu_moments(image_path)
    # zernike_moments = compute_zernike_moments(image_path)
    # Extract SIFT features
    sift_features = extract_sift_features(image_path)

    # # Concatenate Hu Moments and SIFT features
    combined_features = np.concatenate((hu_moments, sift_features))

    return combined_features


def radial_average(image):
    # Get the center of the image
    center = np.array(image.shape) // 2
    Y, X = np.indices(image.shape)
    R = np.sqrt((X - center[1]) ** 2 + (Y - center[0]) ** 2)

    # Flatten arrays
    R_flat = R.flatten()
    image_flat = image.flatten()

    # Define radial bins
    r_max = int(np.ceil(R_flat.max()))
    bin_edges = np.arange(r_max + 2)  # Ensure bin_edges includes the last edge

    # Compute histogram for radial distances
    histogram, _ = np.histogram(R_flat, bins=bin_edges)

    # Compute the sum of magnitudes for each bin
    sum_magnitude, _ = np.histogram(R_flat, bins=bin_edges, weights=image_flat)

    # Compute the average magnitude for each bin
    radial_profile = sum_magnitude / (histogram + 1e-10)  # Avoid division by zero

    return radial_profile


def prepare_data(training_data_path):
    dataset = datasets.ImageFolder(training_data_path, transform=transforms.ToTensor())
    features = []
    labels = []

    for image_path, label in dataset.imgs:
        feature_vector = extract_combined_features(image_path)
        features.append(feature_vector)
        labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    return train_test_split(features, labels, test_size=VAL_FRACTION, random_state=42)


def save_svm_model(model, accuracy, model_dir):
    """
    Save the trained SVM model to a specific directory with a filename including accuracy and timestamp,
    and print the accuracy with a specific tag.

    Parameters:
    - model: The trained SVM model
    - accuracy: Accuracy achieved during validation
    - model_dir: Directory to save the trained model
    - tag: Tag to print the accuracy with
    """
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Generate filename with accuracy and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"SVM_model_{accuracy:.2f}_{timestamp}.pkl"
    model_path = os.path.join(model_dir, filename)

    # Save the model to the specified path
    joblib.dump(model, model_path)

    # Print accuracy with the specific tag
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Model saved to: {model_path}")


def cluster_coordinates(edge_coordinates, num_clusters, debug=False):
    # Check if edge_coordinates is a valid numpy array
    if not isinstance(edge_coordinates, np.ndarray):
        raise ValueError("edge_coordinates must be a numpy array")
    
    num_coords = edge_coordinates.shape[0]
    
    if num_coords < num_clusters:
        # Duplicate coordinates to ensure we have enough data points
        indices = np.random.choice(num_coords, num_clusters, replace=True)
        edge_coordinates = edge_coordinates[indices]
    
    # Perform clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(edge_coordinates)
    cluster_centers = kmeans.cluster_centers_
    
    if debug:
        plt.figure(figsize=(10, 5))
        plt.scatter(edge_coordinates[:, 0], edge_coordinates[:, 1], c='blue', label='Data Points')
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', label='Cluster Centers', marker='x')
        plt.title('KMeans Clustering')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return cluster_centers


def prepare_gcn_graph(coords, k=2, debug=False):
    # Ensure coords is a numpy array
    coords = np.array(coords)

    # Compute k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    # Create edges and calculate distances for edge weights
    edge_index = []
    edge_attr = []
    for i in range(coords.shape[0]):
        for j in indices[i]:
            if i != j:
                edge_index.append([i, j])
                edge_attr.append(distances[i][np.where(indices[i] == j)[0][0]])
    
    # Convert to tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).unsqueeze(1)
    
    # Prepare node features
    x = torch.tensor(coords, dtype=torch.float)
    
    # Create torch_geometric Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    if debug:
        # Create networkx graph for visualization
        G = nx.Graph()
        for i, coord in enumerate(coords):
            G.add_node(i, pos=tuple(coord))
        for edge, weight in zip(edge_index.t().tolist(), edge_attr.tolist()):
            G.add_edge(edge[0], edge[1], weight=weight[0])
        
        # Draw the graph
        pos = nx.get_node_attributes(G, 'pos')
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show()
    
    return data

def main():
    # # Prepare data
    # base_dir = Path('/Users/modeh/EAI2/Type_Dataset')
    # training_data_path = str(base_dir / 'JPEG')
    # model_dir = str(base_dir / 'Model')

    # # Train SVM
    # X_train, X_val, y_train, y_val = prepare_data(training_data_path)
    # svm_model = svm.SVC(kernel='linear',verbose=False)
    # svm_model.fit(X_train, y_train)

    # # Evaluate SVM
    # y_pred = svm_model.predict(X_val)
    # accuracy = accuracy_score(y_val, y_pred)
    # print(f'Validation Accuracy: {accuracy * 100:.2f}%')

    # # Save SVM
    # save_svm_model(svm_model, accuracy, model_dir)
    M2 = "/Users/modeh/EAI2/Metric_Dataset/JPEG/M2/91294A539_Black-Oxide Alloy Steel Hex Drive Flat Head Screw_sim_1_xy_var_1.jpg"
    M3 = "/Users/modeh/EAI2/Metric_Dataset/JPEG/M3/91294A138_Black-Oxide Alloy Steel Hex Drive Flat Head Screw_sim_1_xy_var_0.jpg"
    M4 = "/Users/modeh/EAI2/Metric_Dataset/JPEG/M4/91294A198_Black-Oxide Alloy Steel Hex Drive Flat Head Screw_sim_1_xy_var_1.jpg"
    M5 = "/Users/modeh/EAI2/Metric_Dataset/JPEG/M5/91294A216_Black-Oxide Alloy Steel Hex Drive Flat Head Screw_sim_1_xy_var_1.jpg"
    C2 = "/Users/modeh/EAI2/Type_Dataset/JPEG/Wing_Nut/90866A129_Zinc-Plated Steel Wing Nut_sim_5_xy_var_15.jpg"
    # extract_fft_features([M2,M3,M4,M5],debug=True)
    blurred_image = transformer(M2)
    contour=detect_edges(blurred_image)
    # x=cluster_coordinates(contour,num_clusters=100,debug=True)
    print(contour.shape)
    # prepare_gcn_graph(x)
if __name__ == "__main__":
    main()
