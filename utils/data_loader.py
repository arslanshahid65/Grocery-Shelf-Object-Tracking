import cv2
import os

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return image

def get_image_paths(data_dir):
    return [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]