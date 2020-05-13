import torch
import numpy as np
import time
import copy
import os
import logging
from torchvision import transforms, datasets

input_size = 224
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(filename="log.txt", level=logging.INFO, format=FORMAT)

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train_path': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ColorJitter(hue=.05, saturation=.05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Grayscale(num_output_channels=1),    # convert image to grayscale
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]),
    'val_path': transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.Grayscale(num_output_channels=1),    # convert image to grayscale
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]),
}


def get_class_distribution(dataset_obj):
    count_dict = {k: 0 for k, v in dataset_obj.class_to_idx.items()}

    class_to_idx = dataset_obj.class_to_idx
    idx_to_class = {val: key for (key, val) in class_to_idx.items()}

    for element in dataset_obj:
        y_lbl = element[1]
        y_lbl = idx_to_class[y_lbl]
        count_dict[y_lbl] += 1

    return count_dict


def create_dataloaders(data_dir=None, batch_size=16, weighted_sampler=False):
    print("Initializing Datasets and Dataloaders...")
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train_path', 'val_path']}

    if weighted_sampler:
        target_list = torch.tensor(image_datasets['train_path'].targets)
        target_list = target_list[torch.randperm(len(target_list))]

        class_count = [i for i in get_class_distribution(image_datasets['train_path']).values()]
        class_weights = 1. / torch.tensor(class_count, dtype=torch.float)

        class_weights_all = class_weights[target_list]

        weighted_sampler = torch.utils.data.WeightedRandomSampler(
            weights=class_weights_all,
            num_samples=len(class_weights_all),
            replacement=True
        )

        dataloaders_dict = {}
        dataloaders_dict['train_path'] = torch.utils.data.DataLoader(image_datasets['train_path'],
                                                                     batch_size=batch_size, shuffle=False,
                                                                     num_workers=4,
                                                                     sampler=weighted_sampler)
        dataloaders_dict['val_path'] = torch.utils.data.DataLoader(image_datasets['val_path'], batch_size=batch_size,
                                                                   shuffle=True, num_workers=4)
    else:
        dataloaders_dict = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x
            in ['train_path', 'val_path']}

    return dataloaders_dict


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, is_inception=False, device='cuda',
                save_path=None):

    since = time.time()
    num_classes = len(dataloaders['train_path'].dataset.classes)

    count = 0
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    for epoch in np.arange(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train_path', 'val_path']:
            if phase == 'train_path':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for idx, (inputs, labels) in enumerate(dataloaders[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train_path'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train_path':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train_path':
                        loss.backward()
                        optimizer.step()
                        count += 1

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if count % 100 == 0:
                    print('Iteration {}, Loss {}'.format(count, loss.data))
                    logging.info('Iteration {}, Loss {}'.format(count, loss.data))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            logging.info('{} Epoch:{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val_path' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val_path':
                val_acc_history.append(epoch_acc)

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_name = os.path.join(save_path, 'snapshot_{}.pth.tar'.format(epoch))
        torch.save({
            'epoch': epoch,
            'loss': loss,
            'num_classes': num_classes,
            'class_to_idx': dataloaders['train_path'].dataset.class_to_idx,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, save_name)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}, Epoch: {}'.format(best_acc, best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history