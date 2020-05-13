from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import time
import os
from sklearn.metrics import classification_report

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

test_dir = '/home/golubeva/Projects/mycetozoa/train/test_path'
#weights_path = '/home/golubeva/Projects/mycetozoa/train/snapshots/snapshot_21.pth.tar'
weights_path = '/home/golubeva/Projects/mycetozoa/snapshot_best.pth.tar'

batch_size = 32
input_size = 224
gpu_id = 'cuda:0'
inference_to_folder = False
test_out = '/home/golubeva/Work/furniture/classifier/test_out'


def imshow(inp, path, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imsave(path, inp)
    if title is not None:
        plt.title(title)

start_time = time.time()
############## DATA ##############
data_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.Grayscale(num_output_channels=1),    # convert image to grayscale
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
])

# Create test dataset
image_dataset = datasets.ImageFolder(test_dir, data_transform)

# Create test dataloader
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

device = torch.device(gpu_id)
############## RUN ##############
checkpoint = torch.load(weights_path)

class_to_idx = checkpoint['class_to_idx']
idx_to_class = {val: key for key, val in class_to_idx.items()}
num_classes = len(class_to_idx)
print(len(class_to_idx))

# Initialize the model
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)
model.eval()


fig = plt.figure()
gt, outputs = [], []
running_corrects = 0
with torch.no_grad():
    for idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        preds = model(inputs)
        _, preds_class = torch.max(preds, 1)
        
        if inference_to_folder is True:
            for j in range(inputs.size()[0]):
                label = idx_to_class[int(preds_class[j])]
                out_path = os.path.join(test_out, label, '{}_{}.jpg'.format(idx, j))
                if os.path.exists(os.path.join(test_out, label)) is False:
                    os.mkdir(os.path.join(test_out, label))
                imshow(inputs.cpu().data[j], out_path)
 
        running_corrects += torch.sum(preds_class == labels.data)
        outputs.extend(preds_class.data.cpu().numpy())
        gt.extend(labels.data.cpu().numpy())
        
    
accuracy = running_corrects.double() / len(dataloader.dataset)
print('Acc: {:.4f}'.format(accuracy))
print(classification_report(gt, outputs, target_names=image_dataset.classes))
print("Execution time: --- %s seconds ---" % (time.time() - start_time))
