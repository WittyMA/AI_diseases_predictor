
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- 1. Data Loading ---
# For demonstration, we'll simulate loading a dataset. In a real scenario, you would load one of the datasets
# from the provided links (e.g., Pima Indians Diabetes Dataset).
# You can download the Pima Indians Diabetes Dataset from: https://archive.ics.uci.edu/dataset/34/diabetes
# Make sure to place the 'diabetes.csv' file in the same directory as this script, or provide the full path.

try:
    # Assuming the Pima Indians Diabetes dataset is saved as 'diabetes.csv'
    # Column names are not provided in the original UCI link, so we define them based on common usage.
    # Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv('diabetes.csv', names=column_names)
    print("Dataset loaded successfully.")
    print(data.head())
except FileNotFoundError:
    print("Error: 'diabetes.csv' not found. Please download the Pima Indians Diabetes Dataset from")
    print("https://archive.ics.uci.edu/dataset/34/diabetes and save it as 'diabetes.csv' in the same directory.")
    exit()

# --- 2. Data Preprocessing ---

# Separate features (X) and target (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Handle potential zero values in features where they don't make sense (e.g., BloodPressure, BMI, Glucose)
# These zeros often represent missing values in this specific dataset.
# We'll replace them with the mean of their respective columns.
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_columns:
    X[col] = X[col].replace(0, X[col].mean())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nData preprocessed and split into training and testing sets.")

# --- 3. Model Building (Logistic Regression as an example) ---

# Initialize and train the model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

print("\nModel (Logistic Regression) trained successfully.")

# --- 4. Model Evaluation ---

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # Probability of the positive class

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

print("\nThis script demonstrates a basic workflow for disease prediction. You can replace LogisticRegression")
print("with other models like RandomForestClassifier, SVC, or GradientBoostingClassifier from scikit-learn")
print("and experiment with different preprocessing steps and hyperparameter tuning.")

# Example of how to use a different model (uncomment to try):
# from sklearn.ensemble import RandomForestClassifier
# rf_model = RandomForestClassifier(random_state=42)
# rf_model.fit(X_train_scaled, y_train)
# rf_y_pred = rf_model.predict(X_test_scaled)
# rf_accuracy = accuracy_score(y_test, rf_y_pred)
# print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f}")


