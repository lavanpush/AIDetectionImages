import streamlit as st
import pickle
import numpy as np
import os
from PIL import Image
import logging
from transformers import CLIPProcessor, CLIPModel
from scipy.fftpack import fft2, fftshift
import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# -------------------------
# Load pre-trained models
# -------------------------
@st.cache_resource
def load_models():
    try:
        logging.info("Loading XGBoost model...")
        xgb_model = pickle.load(open("ModelsForUse/xgb_model.pkl", "rb"))
        logging.info("XGBoost model loaded successfully.")

        logging.info("Loading SVM model...")
        svm_model = pickle.load(open("ModelsForUse/svm_model.pkl", "rb"))
        logging.info("SVM model loaded successfully.")

        # Load the scaler used during training
        logging.info("Loading scaler...")
        scaler = pickle.load(open("ModelsForUse/scaler.pkl", "rb"))
        logging.info("Scaler loaded successfully.")

        # Load the neural network model
        logging.info("Loading Neural Network model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the Optuna study
        logging.info("Loading Optuna study...")
        optuna_study_path = "ModelsForUse/nn_optuna_study.pkl"
        if not os.path.exists(optuna_study_path):
            logging.error("Optuna study file not found.")
            st.error("Optuna study file not found.")
            raise FileNotFoundError("Optuna study file not found.")

        with open(optuna_study_path, "rb") as f:
            study = pickle.load(f)

        # Get the best trial parameters
        best_params = study.best_trial.params

        # Define the model architecture with the same parameters used during training
        input_size = 3012  # 2500 FFT features + 512 CLIP features
        hidden_sizes = [
            best_params["hidden_size1"],
            best_params["hidden_size2"],
            best_params["hidden_size3"]
        ]
        dropout_rate = best_params["dropout_rate"]
        nn_model = AdvancedNeuralNetwork(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            num_classes=2,
            dropout_rate=dropout_rate
        ).to(device)

        # Load the state dictionary
        nn_model.load_state_dict(torch.load("ModelsForUse/neural_network.pt", map_location=device))
        nn_model.eval()
        logging.info("Neural Network model loaded successfully.")

        return xgb_model, svm_model, nn_model, scaler
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        st.error(f"Error loading models: {e}")
        raise

# -------------------------
# Load CLIP model
# -------------------------
@st.cache_resource
def load_clip_model():
    try:
        logging.info("Loading CLIP model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        logging.info("CLIP model loaded successfully.")
        return clip_model, clip_processor, device
    except Exception as e:
        logging.error(f"Error loading CLIP model: {e}")
        st.error(f"Error loading CLIP model: {e}")
        raise

# -------------------------
# Feature extraction functions
# -------------------------
def extract_fft_features(image):
    try:
        logging.info("Extracting FFT features...")
        image = image.convert("L").resize((256, 256))
        image_array = np.array(image)

        f_transform = fft2(image_array)
        f_transform_shifted = fftshift(f_transform)
        magnitude_spectrum = np.abs(f_transform_shifted)
        rows, cols = image_array.shape
        crow, ccol = rows // 2, cols // 2
        mask_radius = 25  # Ensure this matches training
        mask = np.zeros((rows, cols))
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= mask_radius ** 2
        mask[mask_area] = 1
        magnitude_spectrum_filtered = magnitude_spectrum * mask
        features = magnitude_spectrum_filtered[
            crow - mask_radius:crow + mask_radius,
            ccol - mask_radius:ccol + mask_radius
        ].flatten()
        normalized_features = (features - np.mean(features)) / (np.std(features) or 1)
        logging.info("FFT features extracted successfully.")
        return normalized_features
    except Exception as e:
        logging.error(f"Error extracting FFT features: {e}")
        raise

def extract_clip_features(image, clip_processor, clip_model, device):
    try:
        logging.info("Extracting CLIP features...")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            clip_embedding = clip_model.get_image_features(**inputs).squeeze(0)
        logging.info("CLIP features extracted successfully.")
        return clip_embedding.cpu().numpy()
    except Exception as e:
        logging.error(f"Error extracting CLIP features: {e}")
        raise

# -------------------------
# Neural Network class
# -------------------------
class AdvancedNeuralNetwork(nn.Module):
    """
    Defines an advanced neural network architecture with Batch Normalization and Dropout.
    """
    def __init__(self, input_size, hidden_sizes, num_classes=2, dropout_rate=0.5):
        super(AdvancedNeuralNetwork, self).__init__()
        layers = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# -------------------------
# Prediction function
# -------------------------
def predict_image(image, model_choice, xgb_model, svm_model, nn_model, clip_processor, clip_model, device, scaler):
    try:
        logging.info("Starting prediction process...")
        fft_features = extract_fft_features(image)  # 2500 features
        clip_features = extract_clip_features(image, clip_processor, clip_model, device)  # 512 features
        combined_features = np.concatenate([fft_features, clip_features])  # Total: 3012

        # Normalize features using the scaler from training
        normalized_features = scaler.transform([combined_features])

        if model_choice == "XGBoost":
            logging.info("Using XGBoost model for prediction...")
            pred_proba = xgb_model.predict_proba(normalized_features)[0]
        elif model_choice == "SVM":
            logging.info("Using SVM model for prediction...")
            pred_proba = svm_model.predict_proba(normalized_features)[0]
        elif model_choice == "Neural Network":
            logging.info("Using Neural Network model for prediction...")
            tensor_features = torch.tensor(normalized_features, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits = nn_model(tensor_features)
                pred_proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
        else:
            logging.error("Invalid model choice")
            st.error("Invalid model choice")
            return None

        logging.info("Prediction completed successfully.")
        return pred_proba
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise

# -------------------------
# Streamlit frontend
# -------------------------
st.title("AI-Generated Image Detector")
st.markdown("Upload an image to classify whether it is **Real** or **Fake**.")

# Load models
logging.info("Loading models...")
xgb_model, svm_model, nn_model, scaler = load_models()
clip_model, clip_processor, device = load_clip_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# Model selection
model_choice = st.selectbox("Select a model", ["XGBoost", "SVM", "Neural Network"])

if uploaded_file:
    try:
        logging.info("Processing uploaded file...")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify Image"):
            with st.spinner("Classifying..."):
                logging.info(f"Model selected: {model_choice}")
                pred_proba = predict_image(
                    image=image,
                    model_choice=model_choice,
                    xgb_model=xgb_model,
                    svm_model=svm_model,
                    nn_model=nn_model,
                    clip_processor=clip_processor,
                    clip_model=clip_model,
                    device=device,
                    scaler=scaler
                )
                if pred_proba is not None:
                    result = "Real" if np.argmax(pred_proba) == 0 else "Fake"
                    confidence = pred_proba[np.argmax(pred_proba)] * 100
                    logging.info(f"Prediction: {result}, Confidence: {confidence:.2f}%")
                    st.success(f"Prediction: **{result}**")
                    st.info(f"Confidence: **{confidence:.2f}%**")
    except Exception as e:
        logging.error(f"An error occurred during classification: {e}")
        st.error(f"An error occurred: {e}")
