import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback

image_size = (32, 32)
batch_size = 2

# Create an ImageDataGenerator instance with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,       # Normalize pixel values
    rotation_range=10,    # Random rotation in the range (degrees, 0 to 180)
    width_shift_range=0.025, # Random horizontal shift
    height_shift_range=0.025, # Random vertical shift
    shear_range=0.015,       # Shear transformation
    zoom_range=0.015,        # Random zoom
    # horizontal_flip=True,  # Randomly flip inputs horizontally
    # vertical_flip=True,    # Randomly flip inputs vertically
    fill_mode='nearest'    # Strategy used for filling in newly created pixels
)

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    'training_set',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary', 
    color_mode='grayscale' 
)

# model = Sequential([
#     Flatten(input_shape=(32, 32, 1)),  # Flatten the 32x32 image into a 1D array
#     Dense(32, activation='relu'),      # A single hidden layer with 32 neurons
#     # Dropout(0.1),                      # Dropout layer to reduce overfitting
#     Dense(8, activation='relu'),      # A second hidden layer with 16 neurons
#     Dense(1, activation='sigmoid')     # Output layer for binary classification
# ])

model = Sequential([
    Conv2D(32, (12, 12), activation='relu', input_shape=(32, 32, 1)),
    MaxPooling2D(3, 3),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

class LossThresholdCallback(Callback):
    def __init__(self, threshold):
        super(LossThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('loss')
        if current_loss is not None:
            if current_loss <= self.threshold:
                print(f"\nReached {self.threshold} loss, so stopping training!")
                self.model.stop_training = True

# Use the custom callback
loss_threshold_callback = LossThresholdCallback(threshold=0.07)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, 
          steps_per_epoch=len(train_generator), 
          epochs=1000, 
        callbacks=[loss_threshold_callback])
model.save('letter_classification_model.keras')
