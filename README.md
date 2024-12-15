# AI Detection of Generated Images

By: Nashrah Haque, Lavanya Pushparaj, & Thien an Pham

## Project Overview

The AI Detection of Generated Images project aims to detect AI-generated images by leveraging advanced feature extraction techniques and machine learning models. It integrates synthetic data generation, feature engineering, and model training in a modular structure.


##  Code Documentation

For code details, check out the [official documentation](https://lavanpush.github.io/AIDetectionImages/).

## Directory Structure

The project is structured for clarity, scalability, and maintainability:

```plaintext
AIDetectionImages/
├── docs/                          # Documentation website files
├── frontend/                      # Frontend application
│   └── app.py                     # Streamlit-based frontend code
├── ModelsForUse/                  # Trained models and generated results
├── src/                           # Core Python modules
│   ├── data.py                    # Handles data loading and synthetic image generation
│   ├── features.py                # Feature extraction methods (FFT, CLIP embeddings)
│   ├── main_pipeline.py           # Orchestrates the pipeline workflow
│   ├── models.py                  # Model training and evaluation
│   └── utils.py                   # Utility functions (logging, configuration)
├── Final_Paper.pdf                # Final Paper
├── README.md                      # Project documentation

```

## Features

### Synthetic Data Generation:
- Generates fake images using the Stable Diffusion model.
- Organizes real and fake image datasets for training.

### Feature Extraction:
- Uses FFT to extract frequency domain features.
- CLIP embeddings for semantic image representation.

### Model Training:
- Trains models such as:
  - Support Vector Machines (SVM)
  - XGBoost (with GPU support)
  - Advanced Neural Networks
- Optimizes hyperparameters using Optuna.

### Evaluation:
- Metrics include accuracy, precision, recall, F1-score, AUROC, and confusion matrices.

### Frontend Application:
- A Streamlit-based app for visualizing results and interacting with models.


## Install Dependencies

The project uses the following Python libraries:
- `torch` (PyTorch) for neural networks and GPU support.
- `transformers` for CLIP embeddings.
- `diffusers` for Stable Diffusion.
- `xgboost` for gradient boosting models.
- `optuna` for hyperparameter optimization.
- `scikit-learn` for SVM and metrics.
- `numpy`, `Pillow`, and `tqdm` for data processing.
- `streamlit` for the frontend application.

To install dependencies, run:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    transformers diffusers xgboost optuna scikit-learn numpy Pillow tqdm streamlit
```

## Usage

### 1. Run the Main Pipeline  
Execute the pipeline to train and evaluate models:  
```bash
python src/main_pipeline.py
```

### 2. Launch the Frontend  
Run the Streamlit app to interact with the results:  

```bash
streamlit run frontend/app.py
```

## Modules and Responsibilities

### `src/data.py`  
- **`load_real_images`**: Loads real images into a PyTorch dataset.  
- **`generate_fake_images`**: Generates synthetic images using Stable Diffusion.  
- **`prepare_image_paths_labels`**: Combines real and fake image paths with labels.  

### `src/features.py`  
- **`extract_fft_features`**: Extracts frequency domain features using FFT.  
- **`extract_clip_embedding`**: Computes semantic embeddings using CLIP.  
- **`prepare_combined_features`**: Combines FFT and CLIP features into a single representation.  

### `src/models.py`  
- **`train_and_save_models`**: Trains models (SVM, XGBoost, and Neural Networks).  
- **`evaluate_and_save_metrics`**: Evaluates models and saves performance metrics.  
- **`AdvancedNeuralNetwork`**: Defines a deep learning model with batch normalization and dropout.  
- **`FusedDataset`**: Custom dataset for combined feature representation.  

### `src/utils.py`  
- **`setup_logging`**: Configures logging for debugging and tracking progress.  

### `src/main_pipeline.py`  
Orchestrates the workflow:  
1. Loads real images.  
2. Generates fake images.  
3. Extracts features.  
4. Trains models.  
5. Evaluates performance.  

### `frontend/app.py`  
Provides a user interface for:  
- Visualizing image datasets.  
- Comparing real vs. fake image classifications.  
- Displaying evaluation metrics.  


## Outputs

- **Trained Models**: Saved in `ModelsForUse/`.  
- **Metrics and Logs**: Results and debugging logs are saved in `pipeline.log` and JSON files in `ModelsForUse/`.  


