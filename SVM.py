import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import transforms
from zernike import RZern
import os
from datetime import datetime
import joblib  # Import joblib for saving the models


VAL_FRACTION = 0.2
IMG_WIDTH = 500


# 7 disrciportrs (invarient to rotation and scale, and translation)
def extract_hu_moments(image_path, debug=False):
    # Load and resizethe image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_WIDTH, IMG_WIDTH))

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
    # Load and resizethe image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_WIDTH, IMG_WIDTH))
    image = image / np.max(image)  # Normalize to range [0, 1]

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


# good for intricate detailes images and texture (invarient to rotation and scale, based on blolbs matching)
def extract_sift_features(image_path, debug=False):
    # Load and resizethe image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_WIDTH, IMG_WIDTH))

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
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_WIDTH, IMG_WIDTH))
    features, _ = hog(
        image,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(1, 1),
        visualize=True,
    )
    return features


def extract_lbp_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_WIDTH, IMG_WIDTH))
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, 10), range=(0, 9))
    return hist


def extract_orb_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    if descriptors is None:
        return np.zeros(128)
    return descriptors.flatten()[:128]  # Limit to 128 elements


def extract_fft_features(image_paths, debug=False):
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    all_avg_magnitudes = []

    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMG_WIDTH, IMG_WIDTH))

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


def extract_combined_features(image_path):
    # Extract Hu Moments
    hu_moments = extract_hu_moments(image_path)
    # zernike_moments = compute_zernike_moments(image_path)
    # Extract SIFT features
    sift_features = extract_sift_features(image_path)

    # # Concatenate Hu Moments and SIFT features
    combined_features = np.concatenate((hu_moments, sift_features))

    return combined_features


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


def find_edges_and_corners(image_path):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Step 1: Edge Detection using Canny
    edges = cv2.Canny(image, 100, 200)

    # Step 2: Corner Detection using Shi-Tomasi
    corners = cv2.goodFeaturesToTrack(
        image, maxCorners=100, qualityLevel=0.01, minDistance=10
    )
    corners = np.int0(corners)

    # Create a copy of the image to draw corners
    image_with_corners = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image_with_corners, (x, y), 3, (0, 255, 0), -1)

    # Step 3: Contour Detection
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    image_with_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)

    # Plot the results
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 3, 1)
    plt.imshow(edges, cmap="gray")
    plt.title("Edges Detected")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(image_with_corners)
    plt.title("Corners Detected")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(image_with_contours)
    plt.title("Contours Detected")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


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

    # extract_fft_features([M2,M3,M4,M5],debug=True)

    find_edges_and_corners(M2)


if __name__ == "__main__":
    main()
