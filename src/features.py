import logging
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
from scipy.fftpack import fft2, fftshift


# -------------------------
# FFT Feature Extraction
# -------------------------
def extract_fft_features(image):
    """
        Extracts features from an image using Fast Fourier Transform (FFT).

        This function performs FFT on the input image to convert it from the spatial domain to the frequency domain.
        It then applies a circular mask to filter the low-frequency components and extracts the magnitude spectrum
        of the filtered result. The extracted features are normalized by subtracting the mean and dividing by the
        standard deviation (with a fallback to 1 if the standard deviation is zero).

        Args:
            image (ndarray): A 2D NumPy array representing the grayscale image to extract features from.

        Returns:
            ndarray: A 1D array of normalized FFT features extracted from the image. Returns None if an error occurs.

        Example:
            image = np.random.rand(256, 256)  # Example image.
            features = extract_fft_features(image)
            print(features)  # Prints the normalized FFT features of the image.
        """
    try:
        f_transform = fft2(image)
        f_transform_shifted = fftshift(f_transform)
        magnitude_spectrum = np.abs(f_transform_shifted)
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        mask_radius = 25
        mask = np.zeros((rows, cols))
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= mask_radius ** 2
        mask[mask_area] = 1
        magnitude_spectrum_filtered = magnitude_spectrum * mask
        features = magnitude_spectrum_filtered[
            crow - mask_radius:crow + mask_radius,
            ccol - mask_radius:ccol + mask_radius
        ].flatten()
        return (features - np.mean(features)) / (np.std(features) or 1)
    except Exception as e:
        logging.error(f"Error in FFT feature extraction: {e}")
        return None

# -------------------------
# CLIP Embedding Extraction
# -------------------------
def extract_clip_embedding(image_path, clip_processor, clip_model, device):
    """
        Extracts CLIP (Contrastive Language-Image Pretraining) embeddings from an image.

        This function loads an image from the specified path, processes it using the given CLIP processor,
        and computes the corresponding image embedding using the CLIP model. The model is run in inference mode
        on the specified device (e.g., CPU or GPU). The output is a NumPy array representing the extracted embedding.

        Args:
            image_path (str): Path to the image file from which the embedding is to be extracted.
            clip_processor (transformers.CLIPProcessor): The CLIP processor used to preprocess the image.
            clip_model (transformers.CLIPModel): The pre-trained CLIP model used to extract image features.
            device (torch.device): The device (CPU or GPU) where the model will run.

        Returns:
            numpy.ndarray: A 1D NumPy array representing the CLIP image embedding. Returns None if an error occurs.

        Example:
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            embedding = extract_clip_embedding("image.jpg", clip_processor, clip_model, device="cuda")
            print(embedding)  # Prints the CLIP embedding for the image.
        """
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = clip_model.get_image_features(**inputs)
        return embedding.squeeze(0).cpu().numpy()
    except Exception as e:
        logging.error(f"Error extracting CLIP embedding for {image_path}: {e}")
        return None


# -------------------------
# Dataset Preparation
# -------------------------
def normalize_array(array, target_size):
    """
        Normalizes the size of a 1D array to the specified target size.

        This function adjusts the size of the input array. If the array is longer than the target size,
        it is truncated. If it is shorter, it is padded with zeros at the end to match the target size.
        If the array is already of the target size, it is returned as is.

        Args:
            array (numpy.ndarray): A 1D NumPy array to be normalized.
            target_size (int): The desired size of the array.

        Returns:
            numpy.ndarray: A 1D NumPy array with the size normalized to the target size.

        Example:
            array = np.array([1, 2, 3, 4, 5])
            normalized_array = normalize_array(array, target_size=7)
            print(normalized_array)  # Output: [1 2 3 4 5 0 0]
        """
    if len(array) > target_size:
        return array[:target_size]
    elif len(array) < target_size:
        return np.pad(array, (0, target_size - len(array)), mode='constant')
    return array

def prepare_combined_features(paths_labels, clip_processor, clip_model, device):
    """
        Prepares combined features for a dataset of images using FFT and CLIP embeddings.

        This function processes a list of image paths and their corresponding labels. For each image:
        - It extracts FFT-based features by converting the image to grayscale and resizing it.
        - It extracts CLIP embeddings using a pre-trained CLIP model.
        - Both feature sets are concatenated into a single combined feature vector.

        Any image that fails to process (either due to missing or incorrect features) is logged, and a record of these failures is saved in a file.
        The function returns the combined feature vectors and labels for the entire dataset.

        Args:
            paths_labels (list of tuples): A list of tuples, where each tuple contains:
                - path (str): The file path of the image.
                - label (int): The label corresponding to the image.
            clip_processor (transformers.CLIPProcessor): The CLIP processor used to preprocess the image.
            clip_model (transformers.CLIPModel): The pre-trained CLIP model used to extract image features.
            device (torch.device): The device (CPU or GPU) on which the CLIP model will run.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: A 2D array of combined feature vectors, where each row is a concatenated FFT and CLIP feature vector.
                - np.ndarray: A 1D array of labels corresponding to the images.

        Example:
            paths_labels = [("image1.jpg", 0), ("image2.jpg", 1)]
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            features, labels = prepare_combined_features(paths_labels, clip_processor, clip_model, device="cuda")
            print(features)  # Prints the combined features.
        """
    fft_features_list, labels = [], []
    failed_paths = []

    for path, label in tqdm(paths_labels, desc="Processing images"):
        try:
            image = Image.open(path).convert("L").resize((256, 256))
            fft_features = extract_fft_features(np.array(image))
            if fft_features is None or fft_features.shape[0] != 2500:
                fft_features = normalize_array(np.zeros(2500), target_size=2500)

            clip_features = extract_clip_embedding(path, clip_processor, clip_model, device)
            if clip_features is None or clip_features.shape[0] != 512:
                clip_features = normalize_array(np.zeros(512), target_size=512)

            combined_feature = np.concatenate((fft_features, clip_features))
            fft_features_list.append(combined_feature)
            labels.append(label)
        except Exception as e:
            logging.error(f"Failed to process {path}: {e}")
            failed_paths.append(path)

    if failed_paths:
        logging.warning(f"Failed to process {len(failed_paths)} images. Check failed_paths.log.")
        with open("failed_paths.log", "w") as f:
            f.write("\n".join(failed_paths))

    return np.array(fft_features_list, dtype=np.float32), np.array(labels, dtype=np.int32)
