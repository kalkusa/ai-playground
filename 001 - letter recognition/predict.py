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
    f'{test_set_path}/A_04.png',
    f'{test_set_path}/A_05.png',
    f'{test_set_path}/B_01.png',
    f'{test_set_path}/B_02.png',
    f'{test_set_path}/B_03.png',
    f'{test_set_path}/B_04.png',
    f'{test_set_path}/B_05.png'
]

# Initialize a variable to track incorrect predictions
incorrect_predictions = 0

# Loop through the images and make predictions
for image_path in image_paths:
    print(f'Testing image: {image_path}')
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class = 'A' if prediction[0][0] < 0.5 else 'B'

    # Check if prediction is correct
    correct_label = 'A' if 'A' in image_path else 'B'
    if predicted_class != correct_label:
        incorrect_predictions += 1
    print(f'{image_path}: The image is predicted to be {predicted_class}\n')

# Check if all images were recognized correctly
if incorrect_predictions == 0:
    print("Success - all images recognized properly")
else:
    print(f"Failure - {incorrect_predictions} images not recognized")
