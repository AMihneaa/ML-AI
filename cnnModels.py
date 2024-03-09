from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 224, 224

num_classes = 3

model = Sequential()

# Convolutional layers for feature extraction
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))  # 3 channels for RGB images
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer to prepare for fully connected layers
model.add(Flatten())

# Fully connected layers for classification
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))  # Softmax for multi-class classification

# Compile the model with optimizer, loss function, and metrics
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Data generators for preprocessing and augmentation (optional)
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Define paths to training and validation directories containing labeled images
train_generator = train_datagen.flow_from_directory(
        'path/to/training/data',
        target_size=(img_width, img_height),
        batch_size=32,  # Adjust batch size based on your hardware capabilities
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        'path/to/validation/data',
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical')

# Train the model
model.fit(train_generator,
          epochs=10,  # Adjust number of epochs based on dataset size and validation performance
          validation_data=validation_generator)

# Save the trained model for future use
model.save('image_ratio_detector.h5')