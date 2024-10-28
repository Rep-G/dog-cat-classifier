from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Load the model
base_dir = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(base_dir, "./classifier.h5"))

# Function to predict if an image is a dog or a cat
def predict_image(image_path):
    image = load_img(image_path, target_size=(128, 128))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    if prediction[0][0] > 0.5:
        print("It's a dog!")
    else:
        print("It's a cat!")

# Main function to run the prediction
if __name__ == "__main__":
    image_path = input("Enter the path to your image: ")  # Prompt for image path
    try:
        predict_image(image_path)  # Call the prediction function
    except Exception as e:
        print(f"An error occurred: {e}")