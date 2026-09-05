import os, warnings, datetime, absl.logging

# === suppress warnings and tensorflow logs ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
warnings.filterwarnings("ignore")
absl.logging.set_verbosity(absl.logging.ERROR)


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam


IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
DATASET_DIR = "asl_alphabet_train"


class TimestampLogger(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\nEpoch {epoch+1}/{self.params['epochs']} — {time_str}")


print("Preparing data generators...")
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset='validation',
    shuffle=False
)

np.save("classes.npy", list(train_gen.class_indices.keys()))
print("Class labels saved to classes.npy")

## load MobileNetV2
print("Downloading MobileNetV2 weights (if not cached)...")
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

##build the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

##train the model
print("Starting training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=0,  
    callbacks=[TimestampLogger()]
)


model.save("asl_transfer_model_new.keras")
print("Model saved as asl_transfer_model_new.keras")

"""
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""
