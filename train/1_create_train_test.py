import os
import cv2
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm


data_path = 'data'
images_ext = ['jpg', 'png', 'jpeg', 'tif', 'tiff']
train_path = 'train_path'
val_path = 'val_path'
test_path = 'test_path'
test_size = 0.15    # percentage of data to create a test sample


fnames = []
labels = []
for folder in os.listdir(data_path):
	current_path = os.path.join(data_path, folder)
	for el in os.listdir(current_path):
		fname = os.path.join(current_path, el)
		img = cv2.imread(fname)

		if img is not None and img.size != 0:
			fnames.append(fname)
			labels.append(folder)
		else:
			print(fname)

print(len(fnames))			

X_train, X_test, y_train, y_test = train_test_split(fnames, labels, test_size=test_size, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, stratify=y_train)

for el in tqdm(X_train):
	train_cls = el.split('/')
	dest_train_folder = os.path.join(train_path, train_cls[-2])
	dest_train_name = os.path.join(train_path, train_cls[-2], train_cls[-1])
	if os.path.exists(dest_train_folder) is False:
		os.makedirs(dest_train_folder)
	shutil.copy(el, dest_train_folder)

for el in tqdm(X_val):
	val_cls = el.split('/')
	dest_val_folder = os.path.join(val_path, val_cls[-2])
	dest_val_name = os.path.join(val_path, val_cls[-2], val_cls[-1])
	if os.path.exists(dest_val_folder) is False:
		os.makedirs(dest_val_folder)
	shutil.copy(el, dest_val_folder)

for el in tqdm(X_test):
	test_cls = el.split('/')
	dest_test_folder = os.path.join(test_path, test_cls[-2])
	dest_test_name = os.path.join(test_path, test_cls[-2], test_cls[-1])
	if os.path.exists(dest_test_folder) is False:
		os.makedirs(dest_test_folder)
	shutil.copy(el, dest_test_folder)
