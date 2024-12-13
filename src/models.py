import os
import json
import logging
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.svm import SVC
import xgboost as xgb
import optuna
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


def evaluate_and_save_metrics(model, model_name, X_test, y_test, output_dir, is_nn=False, device='cpu'):
    """
    Evaluate the model on the test set and save evaluation metrics.

    This function evaluates the given model on the provided test set (`X_test` and `y_test`) by calculating
    various performance metrics including accuracy, precision, recall, F1 score, and AUROC. It also generates
    a classification report. The metrics are saved as a JSON file in the specified output directory.

    Args:
        model (sklearn.base.BaseEstimator or torch.nn.Module): The trained model to be evaluated.
        model_name (str): The name of the model, which will be used for saving the metrics.
        X_test (numpy.ndarray or torch.Tensor): The feature matrix for the test set.
        y_test (numpy.ndarray or torch.Tensor): The true labels for the test set.
        output_dir (str): The directory where the evaluation metrics will be saved.
        is_nn (bool, optional): A flag indicating if the model is a neural network. Defaults to False.
        device (str, optional): The device ('cpu' or 'cuda') to use for inference with neural networks. Defaults to 'cpu'.

    Returns:
        None

    Example:
        evaluate_and_save_metrics(model, "MyModel", X_test, y_test, "path/to/output")
        # Evaluates the model and saves the metrics to the specified output directory.
    """
    logging.info(f"Evaluating {model_name}...")
    os.makedirs(output_dir, exist_ok=True)

    # Predict probabilities and labels
    if is_nn:
        model.eval()
        with torch.no_grad():
            test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            logits = model(test_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    else:
        if not hasattr(model, 'predict_proba'):
            logging.error(f"The model {model_name} does not have 'predict_proba' method.")
            return
        probabilities = model.predict_proba(X_test)

    predictions = np.argmax(probabilities, axis=1)

    # Compute metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    try:
        auroc = roc_auc_score(y_test, probabilities[:, 1])
    except ValueError as e:
        logging.error(f"Error calculating AUROC for {model_name}: {e}")
        auroc = None

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auroc": auroc,
        "classification_report": classification_report(y_test, predictions, output_dict=True)
    }

    # Save metrics as JSON
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logging.info(f"Metrics for {model_name} saved to {metrics_path}.")

# -------------------------
# Neural Network Definitions
# -------------------------
class FusedDataset(Dataset):
    """
    Custom Dataset for handling fused features and labels.

    This class is used to create a PyTorch Dataset that combines features and labels, allowing for easy
    handling and loading of the data during model training or evaluation. The dataset stores the input features
    and labels as tensors, which are then accessed by indexing.

    Args:
        features (numpy.ndarray or torch.Tensor): A 2D array or tensor containing the input features for each sample.
        labels (numpy.ndarray or torch.Tensor): A 1D array or tensor containing the labels for each sample.

    Attributes:
        X (torch.Tensor): A tensor containing the input features.
        y (torch.Tensor): A tensor containing the labels.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Retrieves the feature and label pair for the sample at the specified index.

    Example:
        features = np.random.rand(100, 10)  # 100 samples, 10 features each
        labels = np.random.randint(0, 2, size=100)  # Binary labels for each sample
        dataset = FusedDataset(features, labels)
        print(len(dataset))  # Prints the number of samples in the dataset.
        sample = dataset[0]  # Retrieves the first sample's features and label.
    """
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class AdvancedNeuralNetwork(nn.Module):
    """
    Defines an advanced neural network architecture with Batch Normalization and Dropout.
    """
    def __init__(self, input_size, hidden_sizes, num_classes=2, dropout_rate=0.5):
        """
            Initializes the advanced neural network with specified architecture.

            This constructor builds the neural network layers sequentially. For each hidden layer, it adds a
            fully connected layer, followed by Batch Normalization, ReLU activation, and Dropout for regularization.
            The final layer is a fully connected output layer with the specified number of classes.

            Args:
                input_size (int): The number of input features.
                hidden_sizes (list of int): A list specifying the number of neurons in each hidden layer.
                num_classes (int, optional): The number of output classes. Default is 2 (binary classification).
                dropout_rate (float, optional): The dropout rate applied to each hidden layer. Default is 0.5.

            Returns:
                None
            """
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
        """
            Defines the forward pass of the neural network.

            This function passes the input `x` through the sequential layers of the network. The layers consist
            of fully connected layers, Batch Normalization, ReLU activation, and Dropout (for regularization).
            The final output is generated by the last fully connected layer.

            Args:
                x (torch.Tensor): The input tensor that will be passed through the network.

            Returns:
                torch.Tensor: The output of the neural network after passing through all layers.
            """
        return self.network(x)

# -------------------------
# Objective Function for Optuna
# -------------------------
def objective(trial, X_train, X_valid, y_train, y_valid, device):
    """
    Objective function for Optuna to optimize neural network hyperparameters.

    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        X_train (np.ndarray): Training features.
        X_valid (np.ndarray): Validation features.
        y_train (np.ndarray): Training labels.
        y_valid (np.ndarray): Validation labels.
        device (torch.device): Device to run the model on.

    Returns:
        float: Validation accuracy.
    """
    # Hyperparameters to tune
    hidden_size1 = trial.suggest_int("hidden_size1", 512, 2048)
    hidden_size2 = trial.suggest_int("hidden_size2", 256, 1024)
    hidden_size3 = trial.suggest_int("hidden_size3", 128, 512)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.3, 0.7)

    hidden_sizes = [hidden_size1, hidden_size2, hidden_size3]

    # Define the model
    model = AdvancedNeuralNetwork(input_size=X_train.shape[1],
                                  hidden_sizes=hidden_sizes,
                                  num_classes=2,
                                  dropout_rate=dropout_rate).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define number of epochs
    num_epochs = 10  # Reduced for faster optimization
    batch_size = 128

    # Prepare DataLoaders
    train_dataset = FusedDataset(X_train, y_train)
    valid_dataset = FusedDataset(X_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_features.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        # Optionally, log training loss
        # logging.info(f"Trial {trial.number}, Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_features, batch_labels in valid_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                outputs = model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        accuracy = correct / total
        trial.report(accuracy, epoch)
        model.train()

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

# -------------------------
# Model Training
# -------------------------
def train_and_save_models(X_train, X_test, y_train, y_test, output_dir):
    """
        Trains and saves models using SVM, XGBoost, and a Neural Network, with hyperparameter optimization using Optuna.

        This function standardizes the training data, trains three different machine learning models (SVM, XGBoost, and Neural Network)
        using hyperparameter optimization with Optuna, and saves the models and evaluation metrics.
        For each model, the best hyperparameters are determined via Optuna, the model is trained on the training data,
        and performance metrics are computed and saved.

        Args:
            X_train (numpy.ndarray): The feature matrix for the training set.
            X_test (numpy.ndarray): The feature matrix for the test set.
            y_train (numpy.ndarray): The true labels for the training set.
            y_test (numpy.ndarray): The true labels for the test set.
            output_dir (str): Directory where models, metrics, and other outputs will be saved.

        Returns:
            None

        Example:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            train_and_save_models(X_train, X_test, y_train, y_test, "path/to/output")
            # Trains the models and saves the results to the output directory.
        """
    os.makedirs(output_dir, exist_ok=True)

    # Standardize features using a scaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save the scaler for inference
    with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    logging.info("Scaler saved successfully.")

    # -------------------------
    # SVM Training with Optuna
    # -------------------------
    logging.info("Training SVM model with Optuna...")

    def svm_objective(trial):
        param = {
            'C': trial.suggest_loguniform('C', 1e-4, 10.0),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'probability': True  # Ensure probability=True
        }
        svm = SVC(**param)
        svm.fit(X_train, y_train)
        return accuracy_score(y_test, svm.predict(X_test))

    svm_study = optuna.create_study(direction="maximize")
    svm_study.optimize(svm_objective, n_trials=50)

    # Retrieve the best hyperparameters
    best_params = svm_study.best_params
    # Ensure 'probability' is included
    best_params.update({'probability': True})

    best_svm = SVC(**best_params)
    best_svm.fit(X_train, y_train)

    # Save SVM model
    with open(os.path.join(output_dir, 'svm_model.pkl'), 'wb') as f:
        pickle.dump(best_svm, f)
    logging.info("SVM model saved.")

    # Evaluate SVM and save metrics
    evaluate_and_save_metrics(best_svm, "SVM", X_test, y_test, output_dir)

    # -------------------------
    # XGBoost Training with Optuna
    # -------------------------
    logging.info("Training XGBoost model with Optuna...")

    def xgb_objective(trial):
        param = {
            'lambda': trial.suggest_loguniform('lambda', 1e-4, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-4, 10.0),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 0.1),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.3, 1.0),
            'subsample': trial.suggest_uniform('subsample', 0.3, 1.0),
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
        xgb_model = xgb.XGBClassifier(**param)
        xgb_model.fit(X_train, y_train)
        return accuracy_score(y_test, xgb_model.predict(X_test))

    xgb_study = optuna.create_study(direction="maximize")
    xgb_study.optimize(xgb_objective, n_trials=50)

    # Retrieve the best hyperparameters
    best_params = xgb_study.best_params
    # Include fixed parameters
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False
    })

    best_xgb = xgb.XGBClassifier(**best_params)
    best_xgb.fit(X_train, y_train)

    # Save XGBoost model
    with open(os.path.join(output_dir, 'xgb_model.pkl'), 'wb') as f:
        pickle.dump(best_xgb, f)
    logging.info("XGBoost model saved.")

    # Evaluate XGBoost and save metrics
    evaluate_and_save_metrics(best_xgb, "XGBoost", X_test, y_test, output_dir)

    # -------------------------
    # Neural Network Training with Optuna
    # -------------------------
    logging.info("Training Neural Network with Optuna...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Split training data into training and validation for hyperparameter tuning
    X_train_tune, X_valid, y_train_tune, y_valid = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
    )

    # Define the objective function for Optuna
    def objective_optuna(trial):
        return objective(trial, X_train_tune, X_valid, y_train_tune, y_valid, device)

    # Create a study and optimize
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_optuna, n_trials=50)

    logging.info("Best trial:")
    trial = study.best_trial
    logging.info(f"  Value: {trial.value}")
    logging.info("  Params: ")
    for key, value in trial.params.items():
        logging.info(f"    {key}: {value}")

    # Train the final model with the best hyperparameters on the full training data
    hidden_sizes = [
        trial.params["hidden_size1"],
        trial.params["hidden_size2"],
        trial.params["hidden_size3"],
    ]
    learning_rate = trial.params["learning_rate"]
    dropout_rate = trial.params["dropout_rate"]

    # Define the final model
    final_model = AdvancedNeuralNetwork(input_size=X_train.shape[1],
                                        hidden_sizes=hidden_sizes,
                                        num_classes=2,
                                        dropout_rate=dropout_rate).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = 30
    batch_size = 128
    final_model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_features, batch_labels in DataLoader(FusedDataset(X_train, y_train), batch_size=batch_size, shuffle=True):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass
            outputs = final_model(batch_features)
            loss = criterion(outputs, batch_labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_features.size(0)

        epoch_loss = running_loss / len(X_train)
        logging.info(f"Neural Network Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    logging.info("Neural Network training complete.")

    # Save Neural Network model
    torch.save(final_model.state_dict(), os.path.join(output_dir, 'neural_network.pt'))
    logging.info("Neural Network model saved.")

    # Evaluate Neural Network and save metrics
    evaluate_and_save_metrics(final_model, "NeuralNetwork", X_test, y_test, output_dir, is_nn=True, device=device)

    # Optionally, save the Optuna study
    with open(os.path.join(output_dir, 'nn_optuna_study.pkl'), 'wb') as f:
        pickle.dump(study, f)
    logging.info("Optuna study saved.")