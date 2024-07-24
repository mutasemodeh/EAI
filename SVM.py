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


VAL_FRACTION = 0.2
IMG_WIDTH = 2000


# extract edges_coordinates, corners, contours, lines, circles
def analyze_geometry_image(image_path, debug=False):
    # Assuming transformer function is defined elsewhere
    image = transformer(image_path)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.5)

    # Step 1: Edge Detection using Canny
    edges = cv2.Canny(blurred_image, 100, 200)
    coordinates_edges = np.column_stack(np.where(edges > 0))
    # Step 2: Corner Detection using Shi-Tomasi
    corners = cv2.goodFeaturesToTrack(
        blurred_image, maxCorners=100, qualityLevel=0.01, minDistance=10
    )
    corners = np.int0(corners)

    # Step 3: Contour Detection
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Step 4: Line Detection using Hough Line Transform
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10
    )

    # Step 5: Circle Detection using Hough Circle Transform
    circles = cv2.HoughCircles(
        blurred_image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=200,
        param2=50,
        minRadius=10,
        maxRadius=int(IMG_WIDTH / 2),
    )

    if debug:
        # Create a copy of the original image to draw edges
        image_with_edges = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for coordinates_edge in coordinates_edges:
            x, y = coordinates_edge.ravel()
            cv2.circle(image_with_edges, (y, x), 5, (0, 255, 0), -1)

        # Create a copy of the original image to draw corners
        image_with_corners = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(image_with_corners, (x, y), 5, (0, 255, 0), -1)

        # Draw contours on the original image
        image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

        # Create a copy of the original image to draw lines
        image_with_lines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # Create a copy of the original image to draw circles
        image_with_circles = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]
                # Circle center
                cv2.circle(image_with_circles, center, 1, (0, 100, 100), 5)
                # Circle outline
                cv2.circle(image_with_circles, center, radius, (255, 0, 255), 5)

        # Plot the results
        plt.figure(figsize=(20, 12))

        plt.subplot(2, 3, 1)
        plt.imshow(image, cmap="gray")
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.imshow(image_with_edges, cmap="gray")
        plt.title("Edges Detected")
        plt.axis("off")

        # Plot edges coordinates

        plt.subplot(2, 3, 3)
        plt.imshow(image_with_corners)
        plt.title("Corners Detected")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(image_with_lines)
        plt.title("Lines Detected")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.imshow(image_with_circles)
        plt.title("Circles Detected")
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.imshow(image_with_contours)
        plt.title("Contours Detected")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return edges_coordinates, corners, contours, lines, circles


def detect_edges(image_path, debug=False):
    image = transformer(image_path)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.5)
    edges = cv2.Canny(blurred_image, 100, 200)
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

    return edges_coordinates


def detect_corners(image_path, debug=False):
    image = transformer(image_path)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.5)
    corners = cv2.goodFeaturesToTrack(
        blurred_image, maxCorners=100, qualityLevel=0.01, minDistance=10
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


def detect_contours(image_path, debug=False):
    image = transformer(image_path)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.5)
    edges = cv2.Canny(blurred_image, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
        plt.figure(figsize=(10, 5))
        plt.imshow(image_with_contours)
        plt.title("Contours Detected")
        plt.axis("off")
        plt.show()

    return contours


def detect_lines(image_path, debug=False):
    image = transformer(image_path)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.5)
    edges = cv2.Canny(blurred_image, 100, 200)
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


def detect_circles(image_path, debug=False):
    image = transformer(image_path)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.5)
    circles = cv2.HoughCircles(
        blurred_image,
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

def detect_convex_hulls(image_path, debug=False):
    image = transformer(image_path)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.5)
    edges = cv2.Canny(blurred_image, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if debug:
        image_with_hulls = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for contour in contours:
            hull = cv2.convexHull(contour)
            cv2.drawContours(image_with_hulls, [hull], -1, (0, 255, 0), 2)
        plt.figure(figsize=(10, 5))
        plt.imshow(image_with_hulls)
        plt.title("Convex Hulls Detected")
        plt.axis("off")
        plt.show()

    return contours  

def detect_single_object_rotated_bounding_box(image_path, debug=False):
    image = transformer(image_path)  # Preprocess image if necessary
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.5)
    edges = cv2.Canny(blurred_image, 100, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None  # No contours found

    # Assuming the largest contour is the object of interest
    contour = max(contours, key=cv2.contourArea)
    
    # Compute rotated bounding box
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    if debug:
        # Draw rotated bounding box
        image_with_box = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image_with_box = cv2.addWeighted(image_with_box, 0.5, np.zeros_like(image_with_box), 0, 0)
        cv2.drawContours(image_with_box, [box], 0, (0, 255, 0), 2)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(image_with_box)
        plt.title("Rotated Bounding Box for Single Object")
        plt.axis("off")
        plt.show()

    return box  # Return bounding box points

def extract_hu_moments(image_path, debug=False):
    # 7 disrciportrs (invarient to rotation and scale, and translation)

    # Load and resizethe image in grayscale
    image = transformer(image_path)
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


def compute_zernike_moments(image_path, max_order=8):
    # good for rotationaly invarient shape on a desk

    # Load and resizethe image in grayscale
    image = transformer(image_path)
    # Prepare Zernike calculator
    cart = RZern(max_order)
    L, K = image.shape
    ddx = np.linspace(-1.0, 1.0, K)
    ddy = np.linspace(-1.0, 1.0, L)
    xv, yv = np.meshgrid(ddx, ddy)
    cart.make_cart_grid(xv, yv)

    # Flatten image for computation
    image_flat = image.flatten()
    zernike_moments = np.zeros(cart.nk)

    for i in range(cart.nk):
        Phi = cart.eval_grid(np.eye(cart.nk)[:, i], matrix=True)
        zernike_moments[i] = np.sum(image_flat * Phi.flatten())

    return zernike_moments


def extract_sift_features(image_path, debug=False):
    # good for intricate detailes images and texture (invarient to rotation and scale, based on blolbs matching)

    # Load and resizethe image in grayscale
    image = transformer(image_path)

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


def extract_hog_features(image_path):
    image = transformer(image_path)
    features, _ = hog(
        image,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True,
    )
    return features


def extract_lbp_features(image_path):
    # good for structures with textures

    image = transformer(image_path)
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, 10), range=(0, 9))
    return hist


def extract_orb_features(image_path):
    # similar to SIFT but faster

    image = transformer(image_path)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    if descriptors is None:
        return np.zeros(128)
    return descriptors.flatten()[:128]  # Limit to 128 elements


def extract_fft_features(image_paths, debug=False):
    # good for periodic structres and their singutre

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    all_avg_magnitudes = []

    for image_path in image_paths:
        image = transformer(image_path)

        # Compute FFT
        fft = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft)

        # Compute magnitude spectrum
        magnitude_spectrum = np.abs(fft_shifted)

        # Compute radial average of the magnitude spectrum
        avg_magnitude = radial_average(magnitude_spectrum)
        all_avg_magnitudes.append(avg_magnitude)

    if debug:
        plt.figure(figsize=(10, 5))
        for avg_magnitude in all_avg_magnitudes:
            plt.plot(np.log10(avg_magnitude))
        plt.title("Radial Averaged Magnitude Spectrum")
        plt.xlabel("Radius")
        plt.ylabel("Log10 Magnitude")
        plt.legend([f"Image {i+1}" for i in range(len(image_paths))])
        plt.show()

    return all_avg_magnitudes if len(image_paths) > 1 else all_avg_magnitudes[0]


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


def transformer(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_WIDTH, IMG_WIDTH))
    # image = image / np.max(image)  # Normalize to range [0, 1]
    return image


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



def plot_graph_from_coordinates(coordinates, edges=None):
    """
    Constructs a graph from given (x, y) coordinates and plots it.
    
    Args:
    - coordinates (numpy array): Array of shape (n, 2) where n is the number of nodes, and each row is (x, y) coordinate.
    - edges (list of tuples): Optional list of edges where each edge is a tuple (i, j) representing a connection between nodes i and j.
    
    Returns:
    - G (networkx.Graph): The constructed graph.
    """
    # Create a new graph
    G = nx.Graph()
    
    # Add nodes with positions as node attributes
    for idx, (x, y) in enumerate(coordinates):
        G.add_node(idx, pos=(x, y))
    
    # Add edges if provided
    if edges is not None:
        G.add_edges_from(edges)
    
    # Extract positions for plotting
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw the graph
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10, font_color='black')
    plt.title('Graph from Coordinates')
    plt.show()
    
    return G


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
    x=detect_contours(C2, debug=False)
    print(x.flatten)
    # plot_graph_from_coordinates(x)

if __name__ == "__main__":
    main()
