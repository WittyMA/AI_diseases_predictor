from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from PIL import Image
import io
import base64
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

prediction_bp = Blueprint('prediction', __name__)

# Global variables to store loaded models
MODELS = {}
PREPROCESSORS = {}

# Model paths (adjust these paths based on your model locations)
MODEL_PATHS = {
    'diabetes': '/home/ubuntu/models/diabetes_logistic_regression_model.pkl',
    'heart_disease': '/home/ubuntu/models/heart_disease_random_forest_model.pkl',
    'kidney_disease': '/home/ubuntu/models/kidney_disease_gradient_boosting_model.pkl',
    'liver_disease': '/home/ubuntu/models/liver_disease_logistic_regression_model.pkl',
    'breast_cancer': '/home/ubuntu/models/breast_cancer_svm_model.pkl',
    'covid_symptoms': '/home/ubuntu/models/covid19_symptoms_random_forest_model.pkl',
    'pneumonia_image': '/home/ubuntu/models/pneumonia_detection_model.h5',
    'covid_image': '/home/ubuntu/models/covid19_detection_model.h5'
}

PREPROCESSOR_PATHS = {
    'diabetes': '/home/ubuntu/models/diabetes_preprocessor.pkl',
    'heart_disease': '/home/ubuntu/models/heart_disease_preprocessor.pkl',
    'kidney_disease': '/home/ubuntu/models/kidney_disease_preprocessor.pkl',
    'liver_disease': '/home/ubuntu/models/liver_disease_preprocessor.pkl',
    'breast_cancer': '/home/ubuntu/models/breast_cancer_preprocessor.pkl',
    'covid_symptoms': '/home/ubuntu/models/covid19_symptoms_preprocessor.pkl'
}

def load_models():
    """Load all available models and preprocessors"""
    global MODELS, PREPROCESSORS
    
    # Load tabular models
    for disease, path in MODEL_PATHS.items():
        if disease.endswith('_image'):
            continue
        try:
            if os.path.exists(path):
                MODELS[disease] = joblib.load(path)
                logger.info(f"Loaded {disease} model successfully")
            else:
                logger.warning(f"Model file not found: {path}")
        except Exception as e:
            logger.error(f"Error loading {disease} model: {str(e)}")
    
    # Load preprocessors
    for disease, path in PREPROCESSOR_PATHS.items():
        try:
            if os.path.exists(path):
                PREPROCESSORS[disease] = joblib.load(path)
                logger.info(f"Loaded {disease} preprocessor successfully")
            else:
                logger.warning(f"Preprocessor file not found: {path}")
        except Exception as e:
            logger.error(f"Error loading {disease} preprocessor: {str(e)}")
    
    # Load image models
    for disease, path in MODEL_PATHS.items():
        if not disease.endswith('_image'):
            continue
        try:
            if os.path.exists(path):
                MODELS[disease] = tf.keras.models.load_model(path)
                logger.info(f"Loaded {disease} model successfully")
            else:
                logger.warning(f"Model file not found: {path}")
        except Exception as e:
            logger.error(f"Error loading {disease} model: {str(e)}")

# Load models when the module is imported
load_models()

@prediction_bp.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'loaded_models': list(MODELS.keys()),
        'loaded_preprocessors': list(PREPROCESSORS.keys())
    })

@prediction_bp.route('/predict/diabetes', methods=['POST'])
@cross_origin()
def predict_diabetes():
    """Predict diabetes based on patient data"""
    try:
        data = request.get_json()
        
        # Expected features for diabetes prediction
        features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        # Create DataFrame from input data
        input_df = pd.DataFrame([data], columns=features)
        
        # Preprocess the data
        if 'diabetes' in PREPROCESSORS:
            X_processed = PREPROCESSORS['diabetes'].transform(input_df)
        else:
            return jsonify({'error': 'Diabetes preprocessor not available'}), 500
        
        # Make prediction
        if 'diabetes' in MODELS:
            prediction = MODELS['diabetes'].predict(X_processed)[0]
            probability = MODELS['diabetes'].predict_proba(X_processed)[0]
            
            return jsonify({
                'prediction': int(prediction),
                'probability': {
                    'no_diabetes': float(probability[0]),
                    'diabetes': float(probability[1])
                },
                'confidence': float(max(probability)),
                'risk_level': 'High' if prediction == 1 else 'Low'
            })
        else:
            return jsonify({'error': 'Diabetes model not available'}), 500
            
    except Exception as e:
        logger.error(f"Error in diabetes prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@prediction_bp.route('/predict/heart_disease', methods=['POST'])
@cross_origin()
def predict_heart_disease():
    """Predict heart disease based on patient data"""
    try:
        data = request.get_json()
        
        # Expected features for heart disease prediction
        features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        
        # Create DataFrame from input data
        input_df = pd.DataFrame([data], columns=features)
        
        # Preprocess the data
        if 'heart_disease' in PREPROCESSORS:
            X_processed = PREPROCESSORS['heart_disease'].transform(input_df)
        else:
            return jsonify({'error': 'Heart disease preprocessor not available'}), 500
        
        # Make prediction
        if 'heart_disease' in MODELS:
            prediction = MODELS['heart_disease'].predict(X_processed)[0]
            probability = MODELS['heart_disease'].predict_proba(X_processed)[0]
            
            return jsonify({
                'prediction': int(prediction),
                'probability': {
                    'no_heart_disease': float(probability[0]),
                    'heart_disease': float(probability[1])
                },
                'confidence': float(max(probability)),
                'risk_level': 'High' if prediction == 1 else 'Low'
            })
        else:
            return jsonify({'error': 'Heart disease model not available'}), 500
            
    except Exception as e:
        logger.error(f"Error in heart disease prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@prediction_bp.route('/predict/covid_symptoms', methods=['POST'])
@cross_origin()
def predict_covid_symptoms():
    """Predict COVID-19 based on symptoms"""
    try:
        data = request.get_json()
        
        # Expected features for COVID-19 symptoms prediction
        features = ['Age', 'Fever', 'Tiredness', 'Dry_Cough', 'Difficulty_in_Breathing',
                   'Sore_Throat', 'None_Sympton', 'Pains', 'Nasal_Congestion', 
                   'Runny_Nose', 'Diarrhea', 'None_Experiencing', 'Gender', 'Contact']
        
        # Create DataFrame from input data
        input_df = pd.DataFrame([data], columns=features)
        
        # Preprocess the data
        if 'covid_symptoms' in PREPROCESSORS:
            X_processed = PREPROCESSORS['covid_symptoms'].transform(input_df)
        else:
            return jsonify({'error': 'COVID symptoms preprocessor not available'}), 500
        
        # Make prediction
        if 'covid_symptoms' in MODELS:
            prediction = MODELS['covid_symptoms'].predict(X_processed)[0]
            probability = MODELS['covid_symptoms'].predict_proba(X_processed)[0]
            
            return jsonify({
                'prediction': int(prediction),
                'probability': {
                    'no_covid': float(probability[0]),
                    'covid': float(probability[1])
                },
                'confidence': float(max(probability)),
                'risk_level': 'High' if prediction == 1 else 'Low'
            })
        else:
            return jsonify({'error': 'COVID symptoms model not available'}), 500
            
    except Exception as e:
        logger.error(f"Error in COVID symptoms prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@prediction_bp.route('/predict/chest_xray', methods=['POST'])
@cross_origin()
def predict_chest_xray():
    """Predict disease from chest X-ray image"""
    try:
        data = request.get_json()
        
        if 'image' not in data or 'model_type' not in data:
            return jsonify({'error': 'Image and model_type are required'}), 400
        
        model_type = data['model_type']  # 'pneumonia' or 'covid'
        image_data = data['image']
        
        # Decode base64 image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image based on model requirements
        if model_type == 'pneumonia':
            target_size = (150, 150)
            model_key = 'pneumonia_image'
        elif model_type == 'covid':
            target_size = (224, 224)
            model_key = 'covid_image'
        else:
            return jsonify({'error': 'Invalid model_type. Use "pneumonia" or "covid"'}), 400
        
        image = image.resize(target_size)
        
        # Convert to numpy array and normalize
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        if model_key in MODELS:
            prediction = MODELS[model_key].predict(img_array)[0]
            
            if model_type == 'pneumonia':
                # Binary classification for pneumonia
                confidence = float(prediction[0])
                predicted_class = 'PNEUMONIA' if confidence > 0.5 else 'NORMAL'
                
                return jsonify({
                    'prediction': predicted_class,
                    'confidence': confidence if predicted_class == 'PNEUMONIA' else 1 - confidence,
                    'probabilities': {
                        'NORMAL': float(1 - confidence),
                        'PNEUMONIA': float(confidence)
                    }
                })
            
            elif model_type == 'covid':
                # Multi-class classification for COVID
                class_names = ['COVID', 'NORMAL', 'VIRAL_PNEUMONIA']
                predicted_class_idx = np.argmax(prediction)
                predicted_class = class_names[predicted_class_idx]
                confidence = float(prediction[predicted_class_idx])
                
                probabilities = {class_names[i]: float(prediction[i]) for i in range(len(class_names))}
                
                return jsonify({
                    'prediction': predicted_class,
                    'confidence': confidence,
                    'probabilities': probabilities
                })
        else:
            return jsonify({'error': f'{model_type} model not available'}), 500
            
    except Exception as e:
        logger.error(f"Error in chest X-ray prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@prediction_bp.route('/predict/multimodal', methods=['POST'])
@cross_origin()
def predict_multimodal():
    """Predict using multiple modalities (tabular + image)"""
    try:
        data = request.get_json()
        
        # This is a placeholder for multimodal prediction
        # In a real implementation, you would:
        # 1. Extract features from tabular data using individual models
        # 2. Extract features from image data using CNN models
        # 3. Combine features and use a unified model for final prediction
        
        return jsonify({
            'message': 'Multimodal prediction endpoint',
            'status': 'This feature requires a trained unified multimodal model',
            'suggestion': 'Use individual prediction endpoints for now'
        })
        
    except Exception as e:
        logger.error(f"Error in multimodal prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@prediction_bp.route('/models/status', methods=['GET'])
@cross_origin()
def models_status():
    """Get status of all loaded models"""
    return jsonify({
        'tabular_models': {
            'diabetes': 'diabetes' in MODELS,
            'heart_disease': 'heart_disease' in MODELS,
            'kidney_disease': 'kidney_disease' in MODELS,
            'liver_disease': 'liver_disease' in MODELS,
            'breast_cancer': 'breast_cancer' in MODELS,
            'covid_symptoms': 'covid_symptoms' in MODELS
        },
        'image_models': {
            'pneumonia': 'pneumonia_image' in MODELS,
            'covid': 'covid_image' in MODELS
        },
        'preprocessors': list(PREPROCESSORS.keys())
    })

