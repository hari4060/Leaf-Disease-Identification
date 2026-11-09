import os
import random
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam

# -----------------------
# PARAMETERS
# -----------------------
data_dir = "data"  # Update with your data directory
input_shape = (50, 50, 3)
test_size = 0.2
batch_size = 32
epochs = 15

# -----------------------
# LOAD DATA
# -----------------------
subfolders = os.listdir(data_dir)
data = []

for cls in subfolders:
    cls_dir = os.path.join(data_dir, cls)
    for img_name in os.listdir(cls_dir):
        img_path = os.path.join(cls_dir, img_name)
        data.append((img_path, cls))

train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
train_df = pd.DataFrame(train_data, columns=["image_path", "class"])
test_df = pd.DataFrame(test_data, columns=["image_path", "class"])

# -----------------------
# IMAGE PREPROCESSING
# -----------------------
def preprocess_image(img_path):
    try:
        img_array = cv2.imread(img_path, 1)
        img_array = cv2.medianBlur(img_array, 1)
        new_array = cv2.resize(img_array, input_shape[:2])
        return new_array
    except:
        return None

def prepare_dataset(df):
    X_list, y_list = [], []
    for _, row in df.iterrows():
        img = preprocess_image(row["image_path"])
        if img is not None:
            X_list.append(img)
            y_list.append(subfolders.index(row["class"]))
    X_array = np.array(X_list, dtype=np.float32) / 255.0  # normalize
    y_array = np.array(y_list)
    return X_array, y_array

X_train, y_train = prepare_dataset(train_df)
X_test, y_test = prepare_dataset(test_df)

print("Image preprocessing completed.")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# -----------------------
# BUILD MODEL
# -----------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(len(subfolders), activation='softmax')  # number of classes dynamically
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# -----------------------
# TRAIN MODEL
# -----------------------
history = model.fit(
    X_train, y_train,
    batch_size=16,
    epochs=20,
    validation_data=(X_test, y_test)
)

train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
print(f'Training accuracy: {train_acc:.4f}')

# -----------------------
# SAVE MODEL
# -----------------------
model.save('CNN_model.keras')
print("Saved model to disk")

# -----------------------
# PLOT TRAINING HISTORY
# -----------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# -----------------------
# PREDICTIONS AND EVALUATION
# -----------------------
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

acc = accuracy_score(y_test, y_pred) * 100
print(f"Test Accuracy: {acc:.2f}%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_mtx = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 8))
sns.heatmap(conf_mtx, annot=True, fmt='d', cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

