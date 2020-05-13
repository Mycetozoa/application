from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from train.train_utils import train_model, create_dataloaders

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

data_dir = '/home/golubeva/Projects/mycetozoa/train'
save_path = '/home/golubeva/Projects/mycetozoa/train/snapshots'
gpu_id = 'cuda:0'

## Training options ##
batch_size = 16
num_epochs = 50
lr = 0.001 #0.0001
weight_decay = 0
optimizer_type = 'sgd'

# Detect if we have a GPU available
if torch.cuda.is_available():
    device = torch.device(gpu_id)
else:
    device = torch.device("cpu")

# Create dataloaders
dataloaders_dict = create_dataloaders(data_dir=data_dir, batch_size=batch_size, weighted_sampler=True)
num_classes = len(dataloaders_dict['train_path'].dataset.classes)
print('num_classes:', num_classes)

# Initialize the model
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Send the model to GPU
model = model.to(device)

# optimizer
if optimizer_type == 'adam':
    optimizer_ft = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
elif optimizer_type == 'sgd':
    optimizer_ft = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, patience=5, verbose=True, factor=0.5, mode='max')

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

############ Train and evaluate
model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer_ft, scheduler, num_epochs=num_epochs,
                             is_inception=False, device=device, save_path=save_path)
