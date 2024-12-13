import os
import logging
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import CLIPProcessor, CLIPModel
from utils import setup_logging
from data import load_real_images, generate_fake_images, prepare_image_paths_labels
from features import prepare_combined_features
from models import train_and_save_models


# -------------------------
# Main Function
# -------------------------
def main():
    """
        Main function to run the complete pipeline for image processing, feature extraction,
        model training, and saving results.

        This function orchestrates the following tasks:
        1. Setup logging for monitoring the pipeline.
        2. Load real images from a specified directory and preprocess them.
        3. Generate fake images using a pre-trained Stable Diffusion model.
        4. Prepare the dataset by combining real and fake image paths and labels.
        5. Perform feature extraction using the CLIP model.
        6. Save the extracted features and labels for reuse.
        7. Split the dataset into training and testing sets.
        8. Train models (SVM, XGBoost, Neural Network) with hyperparameter optimization using Optuna.
        9. Save the trained models and evaluation metrics.

        The final pipeline ensures that the models and metrics are stored in a specified output directory
        for future use and evaluation.

        Returns:
            None

        Example:
            main()  # Runs the entire image processing and model training pipeline.
        """
    setup_logging()

    # Configuration
    data_dir = 'extracted_imagenet10'
    fake_images_dir = 'generated_fake_images'
    class_index_json = 'in100_class_index.json'
    output_model_dir = 'ModelsForUse'
    selected_classes = [
        'n02342885', 'n01882714', 'n02129604', 'n03627232',
        'n02980441', 'n02007558', 'n03384352', 'n02279972',
        'n03388043', 'n02391049'
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load Real Images
    real_dataset = load_real_images(data_dir, selected_classes, max_images_per_class=1000)
    logging.info(f"Real dataset loaded with {len(real_dataset)} images.")

    # Generate Fake Images
    generate_fake_images(selected_classes, class_index_json, fake_images_dir, max_images_per_class=1000, device=device)
    logging.info("Fake images generated.")

    # Check Dataset
    for cls in selected_classes:
        class_dir = os.path.join(fake_images_dir, cls)
        if os.path.exists(class_dir):
            fake_count = len([img for img in os.listdir(class_dir) if img.endswith(".png")])
            logging.info(f"Class {cls}: {fake_count} fake images.")

    # Prepare Dataset
    paths_labels = prepare_image_paths_labels(real_dataset, fake_images_dir, selected_classes)
    logging.info(f"Total images prepared (real + fake): {len(paths_labels)}")

    # Feature Extraction
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    features, labels = prepare_combined_features(paths_labels, clip_processor, clip_model, device)
    logging.info(f"Features extracted: {features.shape[0]} samples, {features.shape[1]} features.")

    # Save Features for Reuse
    os.makedirs(output_model_dir, exist_ok=True)
    np.save(os.path.join(output_model_dir, "features.npy"), features)
    np.save(os.path.join(output_model_dir, "labels.npy"), labels)
    logging.info("Features and labels saved for reuse.")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)
    logging.info(f"Data split into training ({len(X_train)}) and testing ({len(X_test)}) samples.")

    # Train and Save Models
    train_and_save_models(X_train, X_test, y_train, y_test, output_model_dir)

    # Final Pipeline Summary
    logging.info("Pipeline completed successfully.")
    logging.info(f"Total images processed: {len(paths_labels)}")
    logging.info(f"Features generated: {features.shape}, Labels generated: {labels.shape}")
    logging.info(f"Models and metrics saved in directory: {output_model_dir}")

if __name__ == "__main__":
    main()
