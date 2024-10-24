import cv2
import matplotlib.pyplot as plt
from models.object_detection import ObjectDetectionModel
from utils.data_loader import load_image, get_image_paths

def main():
    model = ObjectDetectionModel()
    image_dir = 'data/images/'
    image_paths = get_image_paths(image_dir)

    for image_path in image_paths:
        image = load_image(image_path)
        prediction = model.predict(image)

        # Display the image and prediction
        plt.imshow(image)
        plt.title(f'Predicted: {prediction}')
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    main()