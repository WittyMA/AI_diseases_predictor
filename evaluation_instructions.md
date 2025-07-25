## Evaluation Instructions for the Multi-Modal ML/DL System

This document provides instructions on how to evaluate the performance of the multi-modal Automated Disease Prediction System. The evaluation process involves running the provided code for tabular models, image models, and multi-modal integration after preparing the datasets.

### Prerequisites

1.  **Datasets:** Ensure all required datasets (tabular and image) are downloaded and preprocessed according to the `data_acquisition_instructions.md` document. Place them in the recommended directory structure (e.g., `data/processed/`).
2.  **Python Environment:** Make sure you have all necessary libraries installed. You can install them using `pip`:
    ```bash
    pip install pandas scikit-learn tensorflow matplotlib
    ```

### Evaluation Steps

Follow these steps to train and evaluate the individual models and then the integrated multi-modal system:

#### Step 1: Train and Evaluate Tabular Models

1.  Open the `tabular_models.py` file.
2.  Uncomment the example usage blocks for each disease you want to train (Diabetes, Heart Disease, Kidney Disease, Liver Disease, Breast Cancer, COVID-19 Symptoms).
3.  **Crucially, ensure the `data_path` for each model points to your preprocessed CSV file.** For example, if you downloaded and preprocessed the Pima Indians Diabetes dataset and saved it as `data/processed/diabetes.csv`, the line should be:
    ```python
    data_path_diabetes = 'data/processed/diabetes.csv'
    ```
4.  Run the script from your terminal:
    ```bash
    python tabular_models.py
    ```
    This will train and evaluate a model for each uncommented section and print their respective metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC).

#### Step 2: Train and Evaluate Image Models

1.  Open the `image_models.py` file.
2.  Uncomment the example usage blocks for the image datasets you want to train (Pneumonia Detection, COVID-19 Detection, or Multi-class Chest X-Ray Classification).
3.  **Ensure the `train_dir` and `val_dir` paths point to your organized image dataset directories.** For example:
    ```python
    train_dir_pneumonia = 'data/chest_xray_pneumonia/train'
    val_dir_pneumonia = 'data/chest_xray_pneumonia/val'
    ```
4.  Run the script from your terminal:
    ```bash
    python image_models.py
    ```
    This will train and evaluate the image models. It will also save ROC curves as `roc_curve_image_model.png` and training history plots (e.g., `pneumonia_detection_training_history.png`). The trained models will be saved as `.h5` files (e.g., `pneumonia_detection_model.h5`).

#### Step 3: Implement and Evaluate Multi-Modal Integration

1.  Open the `multimodal_integration.py` file.
2.  **Important:** The current `multimodal_integration.py` uses simulated data and dummy feature extractors. To use your actual trained models, you will need to modify the `load_pretrained_tabular_model()` and `load_pretrained_image_model()` functions.
    *   **For `load_pretrained_tabular_model()`:** Instead of returning a dummy function, load your actual trained tabular model (e.g., using `joblib.load` if you saved it, or re-instantiate and load weights if it's a Keras model). This function should then return a method that takes raw tabular data and outputs the features that would be fed into the unified model (e.g., the output of the last layer before the final prediction layer of your tabular model).
    *   **For `load_pretrained_image_model()`:** Similarly, load your trained image model (e.g., `tf.keras.models.load_model("pneumonia_detection_model.h5")`). This function should return a method that takes image data and outputs the features extracted by the CNN (e.g., the output of the `GlobalAveragePooling2D` layer).
3.  Modify the data loading and feature extraction parts in `multimodal_integration.py` to use your actual preprocessed data and the feature extractors from your loaded models.
4.  Run the script from your terminal:
    ```bash
    python multimodal_integration.py
    ```
    This will train and evaluate the unified multi-modal model, printing its performance metrics.

### Interpreting Results

*   **Accuracy:** Overall correctness of the model.
*   **Precision:** Proportion of positive identifications that were actually correct.
*   **Recall (Sensitivity):** Proportion of actual positives that were identified correctly.
*   **F1-Score:** Harmonic mean of precision and recall.
*   **ROC-AUC:** Measures the model's ability to distinguish between classes. A higher value indicates better performance.

By following these steps, you can train and evaluate each component of the multi-modal system and assess the overall performance of the integrated solution.

