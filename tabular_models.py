
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

def train_and_evaluate_tabular_model(data_path, target_column, numerical_cols, categorical_cols, model_name=	'LogisticRegression'):
    """
    Trains and evaluates a machine learning model for tabular data.

    Args:
        data_path (str): Path to the dataset CSV file.
        target_column (str): Name of the target column.
        numerical_cols (list): List of numerical column names.
        categorical_cols (list): List of categorical column names.
        model_name (str): Name of the model to use (e.g., 'LogisticRegression', 'RandomForest', 'SVM', 'GradientBoosting').

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    print(f"\n--- Processing Dataset: {data_path} ---")
    try:
        data = pd.read_csv(data_path)
        print("Dataset loaded successfully.")
        print(data.head())
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Please ensure the dataset is downloaded and placed correctly.")
        return None

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Define preprocessing steps for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown=	'ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            (	'num	', numerical_transformer, numerical_cols),
            (	'cat	', categorical_transformer, categorical_cols)
        ])

    # Define the model
    if model_name == 	'LogisticRegression	':
        model = LogisticRegression(random_state=42, solver=	'liblinear	')
    elif model_name == 	'RandomForest	':
        model = RandomForestClassifier(random_state=42)
    elif model_name == 	'SVM	':
        model = SVC(probability=True, random_state=42)
    elif model_name == 	'GradientBoosting	':
        model = GradientBoostingClassifier(random_state=42)
    else:
        raise ValueError("Unsupported model_name. Choose from 'LogisticRegression', 'RandomForest', 'SVM', 'GradientBoosting'.")

    # Create a pipeline that combines preprocessing and model training
    pipeline = Pipeline(steps=[
        (	'preprocessor	', preprocessor),
        (	'classifier	', model)
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the model
    print(f"Training {model_name} model...")
    pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        	'accuracy	': accuracy_score(y_test, y_pred),
        	'precision	': precision_score(y_test, y_pred, zero_division=0),
        	'recall	': recall_score(y_test, y_pred, zero_division=0),
        	'f1_score	': f1_score(y_test, y_pred, zero_division=0),
        	'roc_auc	': roc_auc_score(y_test, y_pred_proba)
    }

    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.replace(	'_	', 	' 	').title()}: {value:.4f}")

    return metrics

# --- Example Usage for each disease (assuming datasets are in 'data/processed/' directory) ---

# 1. Diabetes (Pima Indians Diabetes Database)
# Make sure to download and preprocess the dataset as 'diabetes.csv'
# The original dataset might need column renaming or header addition.
# Example: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
print("\n--- Running Diabetes Model ---")
data_path_diabetes = 	'data/processed/diabetes.csv	'
numerical_cols_diabetes = [ 	'Pregnancies	', 	'Glucose	', 	'BloodPressure	', 	'SkinThickness	', 	'Insulin	', 	'BMI	', 	'DiabetesPedigreeFunction	', 	'Age	']
categorical_cols_diabetes = [] # Pima dataset is typically all numerical
target_column_diabetes = 	'Outcome	'

# You might need to handle 0 values in some columns as missing data for Pima Indians Diabetes dataset
# For simplicity, this function assumes clean data. In a real scenario, you'd add a custom transformer for this.

# Example of how to run for Diabetes
# metrics_diabetes = train_and_evaluate_tabular_model(
#     data_path_diabetes, target_column_diabetes, numerical_cols_diabetes, categorical_cols_diabetes,
#     model_name=	'RandomForest	'
# )

# 2. Heart Disease (UCI Heart Disease Dataset)
# Make sure to download and preprocess the dataset as 'heart_disease.csv'
# Example columns: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
print("\n--- Running Heart Disease Model ---")
data_path_heart = 	'data/processed/heart_disease.csv	'
numerical_cols_heart = [ 	'age	', 	'trestbps	', 	'chol	', 	'thalach	', 	'oldpeak	', 	'ca	']
categorical_cols_heart = [ 	'sex	', 	'cp	', 	'fbs	', 	'restecg	', 	'exang	', 	'slope	', 	'thal	']
target_column_heart = 	'target	'

# Example of how to run for Heart Disease
# metrics_heart = train_and_evaluate_tabular_model(
#     data_path_heart, target_column_heart, numerical_cols_heart, categorical_cols_heart,
#     model_name=	'SVM	'
# )

# 3. Chronic Kidney Disease (UCI CKD Dataset)
# Make sure to download and preprocess the dataset as 'kidney_disease.csv'
# This dataset has many missing values and categorical columns that need careful handling.
# Example columns: age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane, classification
print("\n--- Running Chronic Kidney Disease Model ---")
data_path_kidney = 	'data/processed/kidney_disease.csv	'
numerical_cols_kidney = [ 	'age	', 	'bp	', 	'bgr	', 	'bu	', 	'sc	', 	'sod	', 	'pot	', 	'hemo	', 	'pcv	', 	'wc	', 	'rc	']
categorical_cols_kidney = [ 	'rbc	', 	'pc	', 	'pcc	', 	'ba	', 	'htn	', 	'dm	', 	'cad	', 	'appet	', 	'pe	', 	'ane	', 	'sg	', 	'al	', 	'su	'] # sg, al, su are numerical but often treated as categorical/ordinal in some contexts
target_column_kidney = 	'classification	'

# Example of how to run for Kidney Disease
# metrics_kidney = train_and_evaluate_tabular_model(
#     data_path_kidney, target_column_kidney, numerical_cols_kidney, categorical_cols_kidney,
#     model_name=	'GradientBoosting	'
# )

# 4. Liver Disease (Indian Liver Patient Dataset - ILPD)
# Make sure to download and preprocess the dataset as 'liver_disease.csv'
# Example columns: Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio, Dataset
print("\n--- Running Liver Disease Model ---")
data_path_liver = 	'data/processed/liver_disease.csv	'
numerical_cols_liver = [ 	'Age	', 	'Total_Bilirubin	', 	'Direct_Bilirubin	', 	'Alkaline_Phosphotase	', 	'Alamine_Aminotransferase	', 	'Aspartate_Aminotransferase	', 	'Total_Protiens	', 	'Albumin	', 	'Albumin_and_Globulin_Ratio	']
categorical_cols_liver = [ 	'Gender	']
target_column_liver = 	'Dataset	' # 1 for liver patient, 2 for non-liver patient

# Example of how to run for Liver Disease
# metrics_liver = train_and_evaluate_tabular_model(
#     data_path_liver, target_column_liver, numerical_cols_liver, categorical_cols_liver,
#     model_name=	'LogisticRegression	'
# )

# 5. Breast Cancer (Wisconsin Diagnostic Breast Cancer Dataset)
# Make sure to download and preprocess the dataset as 'breast_cancer.csv'
# Example columns: id, diagnosis, mean radius, mean texture, mean perimeter, ...
print("\n--- Running Breast Cancer Model ---")
data_path_breast_cancer = 	'data/processed/breast_cancer.csv	'
numerical_cols_breast_cancer = [
    	'mean radius	', 	'mean texture	', 	'mean perimeter	', 	'mean area	', 	'mean smoothness	',
    	'mean compactness	', 	'mean concavity	', 	'mean concave points	', 	'mean symmetry	', 	'mean fractal dimension	',
    	'radius error	', 	'texture error	', 	'perimeter error	', 	'area error	', 	'smoothness error	',
    	'compactness error	', 	'concavity error	', 	'concave points error	', 	'symmetry error	', 	'fractal dimension error	',
    	'worst radius	', 	'worst texture	', 	'worst perimeter	', 	'worst area	', 	'worst smoothness	',
    	'worst compactness	', 	'worst concavity	', 	'worst concave points	', 	'worst symmetry	', 	'worst fractal dimension	'
]
categorical_cols_breast_cancer = []
target_column_breast_cancer = 	'diagnosis	' # M for malignant, B for benign

# Example of how to run for Breast Cancer
# metrics_breast_cancer = train_and_evaluate_tabular_model(
#     data_path_breast_cancer, target_column_breast_cancer, numerical_cols_breast_cancer, categorical_cols_breast_cancer,
#     model_name=	'SVM	'
# )

# 6. COVID-19 Symptoms (Kaggle Dataset)
# Make sure to download and preprocess the dataset as 'covid19_symptoms.csv'
# Example columns: Fever,Tiredness,Dry_Cough,Difficulty_in_Breathing,Sore_Throat,None_Sympton,Pains,Nasal_Congestion,Runny_Nose,Diarrhea,None_Experiencing,Age,Gender,Contact,Infection_Prob
print("\n--- Running COVID-19 Symptoms Model ---")
data_path_covid_symptoms = 	'data/processed/covid19_symptoms.csv	'
numerical_cols_covid_symptoms = [ 	'Age	']
categorical_cols_covid_symptoms = [
    	'Fever	', 	'Tiredness	', 	'Dry_Cough	', 	'Difficulty_in_Breathing	', 	'Sore_Throat	', 	'None_Sympton	',
    	'Pains	', 	'Nasal_Congestion	', 	'Runny_Nose	', 	'Diarrhea	', 	'None_Experiencing	', 	'Gender	', 	'Contact	'
]
target_column_covid_symptoms = 	'Infection_Prob	' # Yes/No or 1/0

# Example of how to run for COVID-19 Symptoms
# metrics_covid_symptoms = train_and_evaluate_tabular_model(
#     data_path_covid_symptoms, target_column_covid_symptoms, numerical_cols_covid_symptoms, categorical_cols_covid_symptoms,
#     model_name=	'RandomForest	'
# )

print("\nTo run these models, uncomment the example usage blocks and ensure the specified dataset paths are correct.")
print("You will need to download and preprocess each dataset according to the instructions in 'data_acquisition_instructions.md'.")


