import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, color_mode='grayscale', target_size=(32, 32))
    img = img_to_array(img)
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Load your trained model
model = load_model('letter_classification_model.keras')

test_set_path = 'test_set'

# List of images to verify
image_paths = [
    f'{test_set_path}/A_01.png',
    f'{test_set_path}/A_02.png',
    f'{test_set_path}/A_03.png',
    f'{test_set_path}/B_01.png',
    f'{test_set_path}/B_02.png',
    f'{test_set_path}/B_03.png'
]

# Loop through the images and make predictions
for image_path in image_paths:
    print(f'Testing image: {image_path}')
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class = 'A' if prediction[0][0] < 0.5 else 'B'
    print(f'{image_path}: The image is predicted to be {predicted_class}\n')
