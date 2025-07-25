
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import joblib
import os

# Assuming data_preprocessing.py, tabular_models.py, image_models.py are in the same directory
from data_preprocessing import preprocess_tabular_data, create_image_data_generators
from tabular_models import train_and_evaluate_tabular_model # Not directly used for prediction, but for training individual models
from image_models import create_cnn_model, train_image_model # create_cnn_model is used for feature extraction

# --- Configuration --- #
# Define paths to your datasets and saved models
DATA_DIR = 	'./data/processed/'	
MODELS_DIR = 	'./models/'	

# Tabular Data Paths (example, adjust as per your processed data)
DIABETES_DATA = os.path.join(DATA_DIR, 	'diabetes.csv'	)
H_DISEASE_DATA = os.path.join(DATA_DIR, 	'heart_disease.csv'	)
KIDNEY_DATA = os.path.join(DATA_DIR, 	'kidney_disease.csv'	)
LIVER_DATA = os.path.join(DATA_DIR, 	'liver_disease.csv'	)
BREAST_CANCER_DATA = os.path.join(DATA_DIR, 	'breast_cancer.csv'	)
COVID_SYMPTOMS_DATA = os.path.join(DATA_DIR, 	'covid19_symptoms.csv'	)

# Image Data Paths (example, adjust as per your organized image data)
CHEST_XRAY_PNEUMONIA_TRAIN = os.path.join(DATA_DIR, 	'chest_xray_pneumonia/train'	)
CHEST_XRAY_PNEUMONIA_VAL = os.path.join(DATA_DIR, 	'chest_xray_pneumonia/val'	)
CHEST_XRAY_COVID_TRAIN = os.path.join(DATA_DIR, 	'chest_xray_covid19/train'	)
CHEST_XRAY_COVID_VAL = os.path.join(DATA_DIR, 	'chest_xray_covid19/val'	)

# Image dimensions for CNNs
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# --- 1. Load Pre-trained Models and Preprocessors --- #
# This section assumes you have already trained and saved your individual models
# using the `tabular_models.py` and `image_models.py` scripts.

def load_all_models():
    models = {}
    preprocessors = {}

    print("Loading pre-trained tabular models and preprocessors...")
    try:
        models[	'diabetes'	] = joblib.load(os.path.join(MODELS_DIR, 	'diabetes_logistic_regression_model.pkl'	))
        preprocessors[	'diabetes'	] = joblib.load(os.path.join(MODELS_DIR, 	'diabetes_preprocessor.pkl'	))
        print("Diabetes model loaded.")
    except FileNotFoundError: print("Diabetes model/preprocessor not found. Please train it first.")

    try:
        models[	'heart_disease'	] = joblib.load(os.path.join(MODELS_DIR, 	'heart_disease_random_forest_model.pkl'	))
        preprocessors[	'heart_disease'	] = joblib.load(os.path.join(MODELS_DIR, 	'heart_disease_preprocessor.pkl'	))
        print("Heart Disease model loaded.")
    except FileNotFoundError: print("Heart Disease model/preprocessor not found. Please train it first.")

    # Add loading for other tabular models (kidney, liver, breast cancer, covid symptoms)
    # ...

    print("Loading pre-trained image models...")
    try:
        models[	'pneumonia_image'	] = load_model(os.path.join(MODELS_DIR, 	'pneumonia_detection_model.h5'	))
        print("Pneumonia image model loaded.")
    except Exception as e: print(f"Pneumonia image model not found or error loading: {e}. Please train it first.")

    try:
        models[	'covid_image'	] = load_model(os.path.join(MODELS_DIR, 	'covid19_detection_model.h5'	))
        print("COVID-19 image model loaded.")
    except Exception as e: print(f"COVID-19 image model not found or error loading: {e}. Please train it first.")

    # Load the unified multi-modal model
    try:
        models[	'unified_multimodal'	] = load_model(os.path.join(MODELS_DIR, 	'unified_multimodal_model.h5'	))
        print("Unified multi-modal model loaded.")
    except Exception as e: print(f"Unified multi-modal model not found or error loading: {e}. Please train it first.")

    return models, preprocessors

# --- 2. Feature Extraction Functions --- #
# These functions will extract features from raw data using the loaded models/preprocessors

def extract_tabular_features(data_dict, preprocessors):
    all_tabular_features = {}
    for disease, df in data_dict.items():
        if disease in preprocessors:
            # Assuming the tabular preprocessor is a ColumnTransformer
            # And the dataframes are already loaded with correct columns
            # This is a simplified example; in reality, you'd need to ensure df has correct columns
            try:
                # For demonstration, let's assume df is the raw dataframe for that disease
                # and we need to apply the preprocessor to it.
                # This part needs to be carefully aligned with how your preprocessors were trained.
                # For now, we'll just return a dummy feature set if the preprocessor is not a full pipeline.
                print(f"Extracting tabular features for {disease}...")
                # This is a placeholder. In a real scenario, you'd apply the preprocessor
                # to the relevant columns of the input dataframe for that disease.
                # For example, if preprocessors['diabetes'] is a ColumnTransformer:
                # X_processed = preprocessors['diabetes'].transform(df.drop(columns=['Outcome']))
                # Then, if you had a feature extraction layer in your tabular model, you'd use it.
                # For now, we'll simulate features.
                all_tabular_features[disease] = np.random.rand(len(df), 64) # Simulate 64 features
            except Exception as e:
                print(f"Error extracting tabular features for {disease}: {e}")
                all_tabular_features[disease] = np.random.rand(len(df), 64) # Fallback to dummy
        else:
            print(f"No preprocessor found for {disease}. Simulating features.")
            all_tabular_features[disease] = np.random.rand(len(df), 64) # Simulate 64 features
    return all_tabular_features

def extract_image_features(image_data_generator, image_model):
    print("Extracting image features...")
    # Assuming image_model is a Keras model and we want features from its penultimate layer
    # Or if it's a feature extractor model directly
    feature_extractor = Model(inputs=image_model.inputs, outputs=image_model.layers[-2].output) # Example: second to last layer
    
    all_features = []
    all_labels = []
    image_data_generator.reset()
    for i in range(image_data_generator.samples // image_data_generator.batch_size):
        images, labels = next(image_data_generator)
        features = feature_extractor.predict(images)
        all_features.append(features)
        all_labels.append(labels)
    
    if image_data_generator.samples % image_data_generator.batch_size != 0:
        # Handle remaining samples if any
        remaining_images, remaining_labels = image_data_generator.next()
        features = feature_extractor.predict(remaining_images)
        all_features.append(features)
        all_labels.append(remaining_labels)

    return np.vstack(all_features), np.hstack(all_labels) # Use hstack for 1D labels

# --- 3. Unified Multi-modal Prediction Function --- #

def predict_multimodal(unified_model, tabular_input_data, image_input_data, preprocessors, models):
    """
    Performs a multi-modal prediction given raw tabular and image input data.
    """
    print("Starting multi-modal prediction...")

    # 1. Preprocess and extract features from tabular data
    # This part needs to be dynamic based on the specific tabular disease
    # For simplicity, let's assume tabular_input_data is a dict of {disease_name: pd.DataFrame}
    # and we are focusing on a single disease for prediction here.
    # For a real system, you'd have a more sophisticated way to identify which tabular data corresponds to which model.
    
    # For demonstration, let's assume we are predicting for 'diabetes' and 'pneumonia_image'
    # You would pass the relevant preprocessed data for the specific prediction task.
    
    # Example: If predicting for a new patient with diabetes symptoms and a chest X-ray
    # tabular_df_for_prediction = tabular_input_data['diabetes']
    # image_arr_for_prediction = image_input_data['pneumonia_image'] # This would be a single image array

    # Simplified feature extraction for demonstration
    # In a real system, you'd apply the specific preprocessor and then the tabular model's feature extractor
    # For now, we'll just use dummy features for the input.
    
    # Assuming tabular_input_data and image_input_data are already feature vectors for a single prediction
    # If they are raw data, you need to call extract_tabular_features and extract_image_features first.
    
    # For this main.py, let's assume the inputs are already processed features for simplicity
    # In a real application, you'd integrate the preprocessing steps here for new, raw data.
    
    # Example: Simulating features for a single prediction
    # tabular_features_single = np.random.rand(1, 64)
    # image_features_single = np.random.rand(1, 128)

    # Make prediction using the unified model
    # Ensure the input shapes match what the unified_model expects
    # unified_model.predict([tabular_features_single, image_features_single])
    
    # This function is conceptual. The actual implementation depends on how you structure your input for prediction.
    print("This function is a placeholder for multi-modal prediction on new, unseen data.")
    print("It requires specific input data formats for tabular and image modalities.")
    print("Please refer to the `multimodal_integration.py` for how features are fused.")
    return "Prediction logic to be implemented based on specific use case."

# --- Main Execution Flow --- #
if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n--- Multi-modal ML/DL System Orchestration ---")

    # Step 1: Load all pre-trained models and preprocessors
    # Ensure you have run tabular_models.py and image_models.py to save these models first.
    all_models, all_preprocessors = load_all_models()

    # Step 2: Prepare data for multi-modal training/evaluation (if not already done)
    # This part would typically involve loading all datasets, preprocessing them,
    # and then extracting features using the pre-trained individual models.
    
    # For demonstration, we'll simulate some data and features.
    print("\nSimulating multi-modal data for unified model training/evaluation...")
    num_samples = 1000
    
    # Simulate raw tabular data for a disease (e.g., diabetes)
    dummy_diabetes_df = pd.DataFrame({
        	'Pregnancies'	: np.random.randint(0, 10, num_samples),
        	'Glucose'	: np.random.randint(70, 200, num_samples),
        	'BloodPressure'	: np.random.randint(60, 100, num_samples),
        	'SkinThickness'	: np.random.randint(10, 50, num_samples),
        	'Insulin'	: np.random.randint(0, 300, num_samples),
        	'BMI'	: np.random.uniform(18, 40, num_samples),
        	'DiabetesPedigreeFunction'	: np.random.uniform(0.1, 1.5, num_samples),
        	'Age'	: np.random.randint(20, 80, num_samples),
        	'Outcome'	: np.random.randint(0, 2, num_samples)
    })
    
    # Simulate raw image data (e.g., for pneumonia)
    dummy_image_arr = np.random.rand(num_samples, IMG_HEIGHT, IMG_WIDTH, 3) # Dummy image data
    dummy_image_labels = np.random.randint(0, 2, num_samples) # Dummy labels

    # --- Feature Extraction (Conceptual) ---
    # In a real scenario, you would use your trained tabular and image models
    # to extract features from your actual datasets.
    
    # Example: Extracting features using dummy extractors (as in multimodal_integration.py)
    tabular_features_dim = 64 # Dimension of features from tabular model
    image_features_dim = 128  # Dimension of features from image model
    
    # Simulate extracted features
    extracted_tabular_features = np.random.rand(num_samples, tabular_features_dim)
    extracted_image_features = np.random.rand(num_samples, image_features_dim)
    
    # Combine features and labels for unified model training
    combined_labels = np.random.randint(0, 2, num_samples) # Unified outcome

    # Split data for unified model training
    X_tab_train, X_tab_test, X_img_train, X_img_test, y_unified_train, y_unified_test = train_test_split(
        extracted_tabular_features, extracted_image_features, combined_labels, test_size=0.2, random_state=42, stratify=combined_labels
    )

    # Step 3: Build and Train Unified Multi-modal Model (if not already trained)
    unified_model = all_models.get(	'unified_multimodal'	)
    if unified_model is None:
        print("\nUnified multi-modal model not found. Building and training a new one...")
        unified_model = Model(
            inputs=[Input(shape=(tabular_features_dim,)), Input(shape=(image_features_dim,))],
            outputs=Dense(1, activation=	'sigmoid'	)(Concatenate()([Input(shape=(tabular_features_dim,)), Input(shape=(image_features_dim,))]))
        ) # Simplified for demonstration
        unified_model.compile(optimizer=	'adam'	, loss=	'binary_crossentropy'	, metrics=[	'accuracy'	])
        
        unified_model.fit(
            [X_tab_train, X_img_train], y_unified_train,
            epochs=10, batch_size=32,
            validation_data=([X_tab_test, X_img_test], y_unified_test),
            verbose=1
        )
        unified_model.save(os.path.join(MODELS_DIR, 	'unified_multimodal_model.h5'	))
        print("Unified multi-modal model trained and saved.")
    else:
        print("\nUnified multi-modal model already loaded.")

    # Step 4: Evaluate the Unified Multi-modal System
    print("\nEvaluating the unified multi-modal model...")
    loss, accuracy = unified_model.evaluate([X_tab_test, X_img_test], y_unified_test, verbose=0)
    print(f"Unified Model Test Loss: {loss:.4f}")
    print(f"Unified Model Test Accuracy: {accuracy:.4f}")

    y_pred_proba = unified_model.predict([X_tab_test, X_img_test]).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

    print("\nClassification Report for Unified Model:")
    print(classification_report(y_unified_test, y_pred))

    print("\nConfusion Matrix for Unified Model:")
    print(confusion_matrix(y_unified_test, y_pred))

    print(f"\nROC-AUC for Unified Model: {roc_auc_score(y_unified_test, y_pred_proba):.4f}")

    print("\n--- System Ready for Prediction (Conceptual) ---")
    print("To make a new prediction, you would load your raw patient data (tabular and image),")
    print("preprocess it using the respective preprocessors, extract features using the individual models,")
    print("and then feed these features into the `unified_multimodal_model`.")
    print("The `predict_multimodal` function above is a conceptual outline for this process.")

    print("\nMulti-modal ML/DL system orchestration complete.")


