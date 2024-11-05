import numpy as np
from scipy.fftpack import fft2, fftshift
from skimage.io import imread
from skimage.transform import resize


# Load and preprocess image
def load_image(image_path):
    image = imread(image_path)
    image = resize(image, (256, 256))  # Resize for consistency
    return image


# Extract FFT features for a single channel
def extract_fft_features(channel, mask_radius=50):
    # Compute the 2D FFT and shift zero frequency to the center
    f_transform = fft2(channel)
    f_transform_shifted = fftshift(f_transform)

    # Convert to magnitude or use real/imaginary parts
    magnitude_spectrum = np.abs(f_transform_shifted)

    # Apply low-pass filter
    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols))
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= mask_radius ** 2
    mask[mask_area] = 1
    magnitude_spectrum_filtered = magnitude_spectrum * mask

    # Flatten and normalize
    flattened_features = magnitude_spectrum_filtered[crow - mask_radius:crow + mask_radius,
                         ccol - mask_radius:ccol + mask_radius].flatten()
    flattened_features = (flattened_features - np.mean(flattened_features)) / np.std(flattened_features)  # Standardize

    return flattened_features


# Extract FFT features for an RGB image
def get_fft_features_for_image(image_path, mask_radius=50):
    image = load_image(image_path)

    # Extract features for each color channel
    red_features = extract_fft_features(image[:, :, 0], mask_radius)
    green_features = extract_fft_features(image[:, :, 1], mask_radius)
    blue_features = extract_fft_features(image[:, :, 2], mask_radius)

    # Concatenate all features into a single vector
    all_features = np.concatenate([red_features, green_features, blue_features])
    return all_features


# Example usage to get FFT features for an image
features = get_fft_features_for_image('/content/realastro.webp')
print("Extracted FFT features shape:", features.shape)
print(features)
