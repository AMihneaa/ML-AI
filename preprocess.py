import cv2
import numpy as np

def preprocess_image(image, target_size=(224, 224)):

    # Convert to RGB format (common for deep learning frameworks)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize image to target size
    image = cv2.resize(image, target_size)

    # Normalize pixel values (assuming uint8 image type)
    image = image.astype(np.float32) / 255.0  # Normalize to range [0, 1]

    return image