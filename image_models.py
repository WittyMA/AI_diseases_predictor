import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import os

def create_cnn_model(input_shape, num_classes=2, model_type=	'simple_cnn	'):
    """
    Creates a CNN model for image classification.

    Args:
        input_shape (tuple): Shape of input images (height, width, channels).
        num_classes (int): Number of output classes.
        model_type (str): Type of model ('simple_cnn', 'vgg16', 'resnet50').

    Returns:
        tf.keras.Model: Compiled CNN model.
    """
    if model_type == 	'simple_cnn	':
        model = Sequential([
            Conv2D(32, (3, 3), activation=	'relu	', input_shape=input_shape),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation=	'relu	'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation=	'relu	'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation=	'relu	'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dropout(0.5),
            Dense(512, activation=	'relu	'),
            Dense(num_classes, activation=	'softmax	' if num_classes > 2 else 	'sigmoid	')
        ])
    elif model_type == 	'vgg16	':
        base_model = VGG16(weights=	'imagenet	', include_top=False, input_shape=input_shape)
        base_model.trainable = False # Freeze the base model
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(128, activation=	'relu	'),
            Dropout(0.5),
            Dense(num_classes, activation=	'softmax	' if num_classes > 2 else 	'sigmoid	')
        ])
    elif model_type == 	'resnet50	':
        base_model = ResNet50(weights=	'imagenet	', include_top=False, input_shape=input_shape)
        base_model.trainable = False # Freeze the base model
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(128, activation=	'relu	'),
            Dropout(0.5),
            Dense(num_classes, activation=	'softmax	' if num_classes > 2 else 	'sigmoid	')
        ])
    else:
        raise ValueError("Unsupported model_type. Choose from 'simple_cnn', 'vgg16', 'resnet50'.")

    # Compile the model
    loss = 	'categorical_crossentropy	' if num_classes > 2 else 	'binary_crossentropy	'
    model.compile(optimizer=Adam(learning_rate=0.001), loss=loss, metrics=[ 	'accuracy	'])

    return model

def train_image_model(train_dir, val_dir, img_height=150, img_width=150, batch_size=32, epochs=15, model_type=	'simple_cnn	', num_classes=2):
    """
    Trains a CNN model for image classification.

    Args:
        train_dir (str): Path to training data directory.
        val_dir (str): Path to validation data directory.
        img_height (int): Height of input images.
        img_width (int): Width of input images.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        model_type (str): Type of model ('simple_cnn', 'vgg16', 'resnet50').
        num_classes (int): Number of output classes.

    Returns:
        tuple: Trained model and training history.
    """
    print(f"\n--- Training {model_type} for Image Classification ---")
    print(f"Training directory: {train_dir}")
    print(f"Validation directory: {val_dir}")

    # Check if directories exist
    if not os.path.exists(train_dir):
        print(f"Error: Training directory {train_dir} not found.")
        return None, None
    if not os.path.exists(val_dir):
        print(f"Error: Validation directory {val_dir} not found.")
        return None, None

    # Image Data Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode=	'nearest	'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load data
    class_mode = 	'categorical	' if num_classes > 2 else 	'binary	'
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=class_mode
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=class_mode
    )

    print(f"Found {train_generator.samples} training images belonging to {train_generator.num_classes} classes.")
    print(f"Found {val_generator.samples} validation images belonging to {val_generator.num_classes} classes.")

    # Create model
    input_shape = (img_height, img_width, 3)
    model = create_cnn_model(input_shape, num_classes, model_type)
    print(f"Model created: {model_type}")
    model.summary()

    # Train model
    print("Starting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        verbose=1
    )

    print("Training complete.")
    return model, history

def evaluate_image_model(model, val_generator):
    """
    Evaluates a trained image classification model.

    Args:
        model (tf.keras.Model): Trained model.
        val_generator: Validation data generator.

    Returns:
        dict: Evaluation metrics.
    """
    print("\n--- Evaluating Model ---")

    # Evaluate on validation set
    loss, accuracy = model.evaluate(val_generator, verbose=0)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")

    # Get predictions
    val_generator.reset()
    predictions = model.predict(val_generator, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1) if predictions.shape[1] > 1 else (predictions > 0.5).astype(int).flatten()

    # Get true labels
    true_classes = val_generator.classes

    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=list(val_generator.class_indices.keys())))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_classes, predicted_classes))

    # For binary classification, compute ROC-AUC
    if predictions.shape[1] == 1 or len(np.unique(true_classes)) == 2:
        if predictions.shape[1] > 1:
            y_scores = predictions[:, 1]
        else:
            y_scores = predictions.flatten()
        
        fpr, tpr, _ = roc_curve(true_classes, y_scores)
        roc_auc = auc(fpr, tpr)
        print(f"\nROC-AUC: {roc_auc:.4f}")

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color=	'darkorange	', lw=2, label=f	'ROC curve (AUC = {roc_auc:.2f})	')
        plt.plot([0, 1], [0, 1], color=	'navy	', lw=2, linestyle=	'--	')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel(	'False Positive Rate	')
        plt.ylabel(	'True Positive Rate	')
        plt.title(	'Receiver Operating Characteristic (ROC) Curve	')
        plt.legend(loc=	'lower right	')
        plt.grid(True)
        plt.savefig(	'roc_curve_image_model.png	', dpi=300, bbox_inches=	'tight	')
        plt.show()

    return {
        	'loss	': loss,
        	'accuracy	': accuracy,
        	'predictions	': predictions,
        	'true_classes	': true_classes,
        	'predicted_classes	': predicted_classes
    }

# --- Example Usage for Chest X-Ray Datasets ---

# 1. Pneumonia Detection from Chest X-Rays
# Dataset structure should be:
# data/chest_xray_pneumonia/
# ├── train/
# │   ├── NORMAL/
# │   └── PNEUMONIA/
# └── val/
#     ├── NORMAL/
#     └── PNEUMONIA/

print("\n--- Pneumonia Detection Model ---")
train_dir_pneumonia = 	'data/chest_xray_pneumonia/train	'
val_dir_pneumonia = 	'data/chest_xray_pneumonia/val	'

# Example of how to run for Pneumonia Detection
# model_pneumonia, history_pneumonia = train_image_model(
#     train_dir_pneumonia, val_dir_pneumonia,
#     img_height=150, img_width=150, batch_size=32, epochs=15,
#     model_type=	'vgg16	', num_classes=2
# )

# if model_pneumonia:
#     metrics_pneumonia = evaluate_image_model(model_pneumonia, val_generator)
#     model_pneumonia.save(	'pneumonia_detection_model.h5	')
#     print("Pneumonia detection model saved as 'pneumonia_detection_model.h5'")

# 2. COVID-19 Detection from Chest X-Rays
# Dataset structure should be:
# data/chest_xray_covid19/
# ├── train/
# │   ├── COVID/
# │   ├── NORMAL/
# │   └── VIRAL_PNEUMONIA/ (if available)
# └── val/
#     ├── COVID/
#     ├── NORMAL/
#     └── VIRAL_PNEUMONIA/ (if available)

print("\n--- COVID-19 Detection Model ---")
train_dir_covid = 	'data/chest_xray_covid19/train	'
val_dir_covid = 	'data/chest_xray_covid19/val	'

# For COVID-19 detection, you might have 2 classes (COVID vs NORMAL) or 3 classes (COVID vs NORMAL vs VIRAL_PNEUMONIA)
# Adjust num_classes accordingly

# Example of how to run for COVID-19 Detection (binary classification: COVID vs NORMAL)
# model_covid, history_covid = train_image_model(
#     train_dir_covid, val_dir_covid,
#     img_height=224, img_width=224, batch_size=16, epochs=20,
#     model_type=	'resnet50	', num_classes=2
# )

# if model_covid:
#     metrics_covid = evaluate_image_model(model_covid, val_generator)
#     model_covid.save(	'covid19_detection_model.h5	')
#     print("COVID-19 detection model saved as 'covid19_detection_model.h5'")

# 3. Multi-class Classification (COVID vs NORMAL vs PNEUMONIA)
# If you have a dataset with 3 classes, set num_classes=3

print("\n--- Multi-class Chest X-Ray Classification Model ---")
train_dir_multiclass = 	'data/chest_xray_multiclass/train	'
val_dir_multiclass = 	'data/chest_xray_multiclass/val	'

# Example of how to run for Multi-class Classification
# model_multiclass, history_multiclass = train_image_model(
#     train_dir_multiclass, val_dir_multiclass,
#     img_height=224, img_width=224, batch_size=16, epochs=25,
#     model_type=	'resnet50	', num_classes=3
# )

# if model_multiclass:
#     metrics_multiclass = evaluate_image_model(model_multiclass, val_generator)
#     model_multiclass.save(	'multiclass_chest_xray_model.h5	')
#     print("Multi-class chest X-ray model saved as 'multiclass_chest_xray_model.h5'")

def plot_training_history(history, model_name):
    """
    Plots training and validation accuracy and loss.

    Args:
        history: Training history from model.fit().
        model_name (str): Name of the model for plot title.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot accuracy
    ax1.plot(history.history[	'accuracy	'], label=	'Training Accuracy	')
    ax1.plot(history.history[	'val_accuracy	'], label=	'Validation Accuracy	')
    ax1.set_title(f	'{model_name} - Model Accuracy	')
    ax1.set_xlabel(	'Epoch	')
    ax1.set_ylabel(	'Accuracy	')
    ax1.legend()
    ax1.grid(True)

    # Plot loss
    ax2.plot(history.history[	'loss	'], label=	'Training Loss	')
    ax2.plot(history.history[	'val_loss	'], label=	'Validation Loss	')
    ax2.set_title(f	'{model_name} - Model Loss	')
    ax2.set_xlabel(	'Epoch	')
    ax2.set_ylabel(	'Loss	')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f	'{model_name.lower().replace(" ", "_")}_training_history.png	', dpi=300, bbox_inches=	'tight	')
    plt.show()

# Example usage of plotting function:
# if 'history_pneumonia' in locals() and history_pneumonia:
#     plot_training_history(history_pneumonia, "Pneumonia Detection")

print("\nTo run these image models, uncomment the example usage blocks and ensure the specified dataset paths are correct.")
print("You will need to download and organize the chest X-ray datasets according to the instructions in 'data_acquisition_instructions.md'.")
print("Make sure to create the appropriate directory structure with 'train' and 'val' folders, each containing class subfolders.")

