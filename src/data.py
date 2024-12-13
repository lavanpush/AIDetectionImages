import os
import logging
import torch
from torchvision import transforms as trn
from torchvision.datasets import ImageFolder
from diffusers import StableDiffusionPipeline
import json


# -------------------------
# Load Real Images
# -------------------------
def load_real_images(data_dir, selected_classes, max_images_per_class=1000):
    """
        Loads a subset of real images from a directory, filtering by selected classes and limiting the number of images per class.

        This function loads images from the specified directory, selecting only those belonging to the classes
        provided in `selected_classes`. It ensures that no more than `max_images_per_class` images are loaded
        per class. The images are preprocessed by resizing and centering them to 256x256 pixels and converting
        them to tensor format.

        Args:
            data_dir (str): The directory containing the image data organized into subfolders, where each subfolder
                            corresponds to a class.
            selected_classes (list of str): A list of class names to filter the dataset by.
            max_images_per_class (int, optional): The maximum number of images to load per selected class. Default is 1000.

        Returns:
            torch.utils.data.Subset: A subset of the `ImageFolder` dataset containing the filtered and preprocessed images.

        Example:
            data_dir = "path/to/imagenet"
            selected_classes = ["tiger", "koala", "hamster"]
            dataset = load_real_images(data_dir, selected_classes, max_images_per_class=500)
            print(len(dataset))  # Prints the number of images loaded.
        """
  #Real images taken from imagenet - 1000 images (small subset with 10 classes) which are: hamster, zebra, castle, fountain, koala, tiger, monarch butterfly, flamingo, knot, forklift
    logging.info("Loading real images...")
    transform_real = trn.Compose([
        trn.Resize(256),
        trn.CenterCrop(256),
        trn.ToTensor(),
    ])
    dataset = ImageFolder(root=data_dir, transform=transform_real)
    class_to_idx = dataset.class_to_idx
    selected_class_indices = [class_to_idx[cls] for cls in selected_classes if cls in class_to_idx]
    indices = [i for i, (_, label) in enumerate(dataset.samples) if label in selected_class_indices]

    class_counts = {cls_idx: 0 for cls_idx in selected_class_indices}
    limited_indices = []
    for idx in indices:
        _, label = dataset.samples[idx]
        if class_counts[label] < max_images_per_class:
            limited_indices.append(idx)
            class_counts[label] += 1

    subset_dataset = torch.utils.data.Subset(dataset, limited_indices)
    logging.info(f"Total real images loaded: {len(subset_dataset)}")
    return subset_dataset

# -------------------------
# Generate Fake Images
# -------------------------
def generate_fake_images(selected_classes, class_index_json, output_dir, max_images_per_class=1000, device='cuda'):
    """
        Generates fake images for selected classes using Stable Diffusion.

        This function uses the Stable Diffusion model to generate fake images for the specified classes. It takes
        class identifiers (WNID) from `selected_classes` and retrieves the corresponding class names from a JSON file
        (`class_index_json`). It generates images based on the class names and saves them to the `output_dir`.
        A maximum of `max_images_per_class` images are generated for each class. If images already exist for a class,
        it skips generating additional images.

        Args:
            selected_classes (list of str): A list of class identifiers (WNIDs) for which fake images should be generated.
            class_index_json (str): Path to a JSON file that maps class indices to class names.
            output_dir (str): Directory where the generated images will be saved.
            max_images_per_class (int, optional): The maximum number of images to generate per class. Default is 1000.
            device (str, optional): The device on which the model should run (e.g., 'cuda' or 'cpu'). Default is 'cuda'.

        Returns:
            None

        Example:
            selected_classes = ["n02096585", "n02129604"]
            class_index_json = "path/to/class_index.json"
            output_dir = "path/to/output"
            generate_fake_images(selected_classes, class_index_json, output_dir)
            # Fake images for the selected classes are generated and saved in the output directory.
        """

    #generates fake images using stable diffusion
    logging.info("Checking existing fake images...")
    os.makedirs(output_dir, exist_ok=True)

    with open(class_index_json, 'r') as f:
        class_index = json.load(f)

    wnid_to_class_name = {value[1]: key for key, value in class_index.items()}
    sd_pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
    sd_pipeline.safety_checker = lambda images, **kwargs: (images, [False] * len(images))

    for wnid in selected_classes:
        if wnid not in wnid_to_class_name:
            logging.warning(f"WNID {wnid} not found in class index. Skipping.")
            continue

        class_name = wnid_to_class_name[wnid]
        prompt = class_name.replace('_', ' ')
        class_dir = os.path.join(output_dir, wnid)
        os.makedirs(class_dir, exist_ok=True)

        existing_images = [img for img in os.listdir(class_dir) if img.endswith('.png')]
        if len(existing_images) >= max_images_per_class:
            logging.info(f"Images for class {class_name} already exist. Skipping.")
            continue

        logging.info(f"Generating images for class: {class_name} (WNID: {wnid})")
        for idx in range(max_images_per_class - len(existing_images)):
            try:
                image = sd_pipeline(prompt).images[0]
                image.save(os.path.join(class_dir, f"{wnid}_{len(existing_images) + idx:05d}.png"))
            except Exception as e:
                logging.error(f"Failed to generate image for {class_name}, index {idx}: {e}")

    logging.info("Fake image generation complete.")

# -------------------------
# Prepare Real and Fake Datasets
# -------------------------
def prepare_image_paths_labels(real_dataset, fake_dir, selected_classes):
    """
        Prepares a combined list of image paths and labels from real and fake datasets.

        This function creates a list of image paths and their corresponding labels for both real and fake images.
        For real images, the paths and labels are obtained from the `real_dataset`. For fake images, the paths
        are gathered from the specified `fake_dir` for each class in `selected_classes`. The real images are labeled as 0,
        and the fake images are labeled as 1. The function returns a combined list of tuples containing image paths
        and their associated labels.

        Args:
            real_dataset (torch.utils.data.Subset): A subset of the real image dataset, typically with transformations applied.
            fake_dir (str): Directory where fake images are stored, organized by class.
            selected_classes (list of str): A list of selected classes for which fake images should be included.

        Returns:
            list of tuples: A list of tuples, where each tuple contains:
                - str: The file path to an image.
                - int: The label for the image (0 for real, 1 for fake).

        Example:
            real_dataset = load_real_images("path/to/real_images", selected_classes=["hamster", "zebra"])
            fake_dir = "path/to/fake_images"
            selected_classes = ["hamster", "zebra"]
            image_paths_labels = prepare_image_paths_labels(real_dataset, fake_dir, selected_classes)
            print(image_paths_labels)  # Prints the list of image paths with labels.
        """
    real_image_paths_labels = [(real_dataset.dataset.samples[idx][0], 0) for idx in real_dataset.indices]
    fake_image_paths_labels = []
    for cls in selected_classes:
        class_dir = os.path.join(fake_dir, cls)
        if not os.path.exists(class_dir):
            continue
        fake_images = [os.path.join(class_dir, img_name) for img_name in os.listdir(class_dir) if img_name.endswith('.png')]
        fake_image_paths_labels.extend([(img_path, 1) for img_path in fake_images])
    return real_image_paths_labels + fake_image_paths_labels