
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def preprocess_tabular_data(df, numerical_cols, categorical_cols, target_column=None):
    """
    Preprocesses tabular data by handling missing values, encoding categorical features,
    and scaling numerical features.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numerical_cols (list): List of numerical column names.
        categorical_cols (list): List of categorical column names.
        target_column (str, optional): Name of the target column. If provided, it will be excluded from preprocessing.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
        ColumnTransformer: The fitted preprocessor pipeline.
    """
    print("Starting tabular data preprocessing...")

    # Separate features and target if target_column is provided
    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df.copy()
        y = None

    # Define preprocessing steps
    # Numerical pipeline: Impute missing values with mean, then scale
    numerical_transformer = Pipeline(steps=[
        (	'imputer'	, SimpleImputer(strategy=	'mean'	)),
        (	'scaler'	, StandardScaler())
    ])

    # Categorical pipeline: Impute missing values with most frequent, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        (	'imputer'	, SimpleImputer(strategy=	'most_frequent'	)),
        (	'onehot'	, OneHotEncoder(handle_unknown=	'ignore'	))
    ])

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            (	'num'	, numerical_transformer, numerical_cols),
            (	'cat'	, categorical_transformer, categorical_cols)
        ],
        remainder=	'passthrough'	 # Keep other columns (if any) that are not specified
    )

    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)

    # Convert processed data back to DataFrame (optional, for easier inspection)
    # Get feature names after one-hot encoding
    try:
        cat_feature_names = preprocessor.named_transformers_[	'cat'	][	'onehot'	].get_feature_names_out(categorical_cols)
    except AttributeError:
        # For older scikit-learn versions
        cat_feature_names = preprocessor.named_transformers_[	'cat'	][	'onehot'	].get_feature_names(categorical_cols)

    all_feature_names = numerical_cols + list(cat_feature_names)

    X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names, index=df.index)

    print("Tabular data preprocessing complete.")
    if y is not None:
        return X_processed_df, y, preprocessor
    else:
        return X_processed_df, preprocessor

def create_image_data_generators(train_dir, val_dir, img_height, img_width, batch_size):
    """
    Creates and returns image data generators for training and validation.

    Args:
        train_dir (str): Path to the training data directory.
        val_dir (str): Path to the validation data directory.
        img_height (int): Desired image height.
        img_width (int): Desired image width.
        batch_size (int): Batch size for generators.

    Returns:
        tuple: (train_generator, validation_generator)
    """
    print("Setting up image data generators...")

    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found at {train_dir}")
        return None, None
    if not os.path.exists(val_dir):
        print(f"Error: Validation directory not found at {val_dir}")
        return None, None

    train_datagen = ImageDataGenerator(
        rescale=1./255,          # Normalize pixel values
        rotation_range=20,       # Randomly rotate images
        width_shift_range=0.2,   # Randomly shift image width
        height_shift_range=0.2,  # Randomly shift image height
        shear_range=0.2,         # Apply shear transformation
        zoom_range=0.2,          # Apply random zoom
        horizontal_flip=True,    # Randomly flip images horizontally
        fill_mode=	'nearest'	      # Fill newly created pixels
    )

    val_datagen = ImageDataGenerator(rescale=1./255) # Only rescale for validation

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=	'binary'	 # Assuming binary classification for simplicity (e.g., Normal/Pneumonia)
    )

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=	'binary'	
    )

    print("Image data generators created.")
    return train_generator, validation_generator

# Example Usage (for demonstration, uncomment to test):
# if __name__ == "__main__":
#     # --- Tabular Data Example ---
#     print("\n--- Tabular Data Preprocessing Example ---")
#     # Create a dummy DataFrame
#     data = {
#         	'Age'	: [25, 30, np.nan, 40, 45],
#         	'Gender'	: [	'Male'	, 	'Female'	, 	'Male'	, 	'Female'	, 	'Male'	],
#         	'Cholesterol'	: [200, 220, 180, np.nan, 250],
#         	'Smoker'	: [	'Yes'	, 	'No'	, 	'No'	, 	'Yes'	, 	'No'	],
#         	'Disease'	: [0, 1, 0, 1, 0]
#     }
#     df = pd.DataFrame(data)

#     numerical_cols = [	'Age'	, 	'Cholesterol'	]
#     categorical_cols = [	'Gender'	, 	'Smoker'	]
#     target_column = 	'Disease'	

#     X_processed_df, y, preprocessor = preprocess_tabular_data(df, numerical_cols, categorical_cols, target_column)
#     print("\nOriginal DataFrame:")
#     print(df)
#     print("\nPreprocessed Features (X_processed_df):")
#     print(X_processed_df)
#     print("\nTarget (y):")
#     print(y)

#     # --- Image Data Example ---
#     print("\n--- Image Data Preprocessing Example ---")
#     # Create dummy directories for demonstration
#     os.makedirs(	'temp_data/train/class_a'	, exist_ok=True)
#     os.makedirs(	'temp_data/train/class_b'	, exist_ok=True)
#     os.makedirs(	'temp_data/val/class_a'	, exist_ok=True)
#     os.makedirs(	'temp_data/val/class_b'	, exist_ok=True)

#     # Create dummy image files (empty files for demonstration)
#     with open(	'temp_data/train/class_a/img1.jpg'	, 	'w'	) as f: pass
#     with open(	'temp_data/train/class_b/img2.jpg'	, 	'w'	) as f: pass
#     with open(	'temp_data/val/class_a/img3.jpg'	, 	'w'	) as f: pass

#     train_dir = 	'temp_data/train'	
#     val_dir = 	'temp_data/val'	
#     img_height, img_width, batch_size = 128, 128, 16

#     train_gen, val_gen = create_image_data_generators(train_dir, val_dir, img_height, img_width, batch_size)

#     if train_gen and val_gen:
#         print(f"Train generator found {train_gen.samples} images.")
#         print(f"Validation generator found {val_gen.samples} images.")

#     # Clean up dummy directories
#     import shutil
#     shutil.rmtree(	'temp_data'	)


