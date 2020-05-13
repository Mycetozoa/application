import io
from PIL import Image
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision import models
import os

weights_path = 'snapshot_best.pth.tar'
gpu_id = "cpu"
input_size = 224

checkpoint = torch.load(weights_path, map_location=gpu_id)
class_to_idx = checkpoint['class_to_idx']
idx_to_class = {val: key for key, val in class_to_idx.items()}
num_classes = len(class_to_idx)

# Initialize the model
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
print('Model loaded')


def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.Grayscale(num_output_channels=1),    # convert image to grayscale
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    try:
        tensor = transform_image(image_bytes=image_bytes)
        with torch.no_grad():
            outputs = model(tensor)
    except Exception:
        return 0, 'error'
    _, preds = torch.max(outputs, 1)
    predicted_idx = preds.item()
    return predicted_idx, idx_to_class[predicted_idx]


if __name__ == '__main__':
    imagePath = '/home/golubeva/Projects/mycetozoa/train/test_path'

    for root, dirs, files in os.walk(imagePath):
        for file in files:
            filepath = os.path.join(root, file)

            with open(filepath, 'rb') as f:
                img_bytes = f.read()

            pred = get_prediction(img_bytes)
            print(pred)
    pass