import torch
import torchvision.transforms as transforms
from torchvision import models

class ObjectDetectionModel:
    def __init__(self):
        # Load a pre-trained ResNet model
        self.model = models.resnet50(pretrained=True)
        self.model.eval()  # Set the model to evaluation mode

        # Load class labels
        with open('data/labels.txt') as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image):
        # Apply transformations
        image = self.transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs.data, 1)

        return self.labels[predicted.item()]