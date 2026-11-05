"""Train a CNN to classify fruit images as fresh or rotten."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix


ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
MODEL_PATH = ROOT_DIR / "model" / "food_freshness.keras"
CLASS_NAMES = ["fresh", "rotten"]
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 20
SEED = 123
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def load_datasets():
    """Load the train, validation, and test folders in a fixed class order."""
    datasets = {}
    for split in ("train", "validation", "test"):
        split_dir = DATASET_DIR / split
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing dataset folder: {class_dir}")
            if not any(
                image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
                for image in class_dir.rglob("*")
            ):
                raise ValueError(f"No images found in {class_dir}")

        datasets[split] = tf.keras.utils.image_dataset_from_directory(
            split_dir,
            class_names=CLASS_NAMES,
            image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="binary",
            shuffle=split == "train",
            seed=SEED,
        ).prefetch(tf.data.AUTOTUNE)

    return datasets["train"], datasets["validation"], datasets["test"]


def build_model():
    """Create the binary fresh/rotten CNN."""
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(*IMAGE_SIZE, 3)),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_accuracy_plot(history):
    """Save training and validation accuracy across epochs."""
    epoch_numbers = range(1, len(history.history["accuracy"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_numbers, history.history["accuracy"], label="Training accuracy")
    plt.plot(
        epoch_numbers,
        history.history["val_accuracy"],
        label="Validation accuracy",
    )
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path = MODEL_PATH.parent / "training_accuracy.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Training accuracy plot saved to {output_path}")


def save_confusion_matrix(true_labels, predicted_labels):
    """Save a labeled confusion matrix for the held-out test images."""
    matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=CLASS_NAMES)
    figure, axis = plt.subplots(figsize=(6, 5))
    display.plot(ax=axis, cmap="Blues", values_format="d", colorbar=False)
    axis.set_title("Test Set Confusion Matrix")
    figure.tight_layout()
    output_path = MODEL_PATH.parent / "confusion_matrix.png"
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Confusion matrix saved to {output_path}")


def main():
    train_dataset, validation_dataset, test_dataset = load_datasets()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    model = build_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
    )
    save_training_accuracy_plot(history)

    best_model = tf.keras.models.load_model(MODEL_PATH)
    test_loss, test_accuracy = best_model.evaluate(test_dataset, verbose=0)
    true_labels = np.concatenate(
        [labels.numpy().reshape(-1) for _, labels in test_dataset]
    ).astype(int)
    rotten_probabilities = best_model.predict(test_dataset, verbose=0).reshape(-1)
    predicted_labels = (rotten_probabilities >= 0.5).astype(int)
    save_confusion_matrix(true_labels, predicted_labels)

    print(f"Test accuracy: {test_accuracy:.2%}")
    print(f"Test loss: {test_loss:.4f}")
    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
