import os
import os.path as osp
from PIL import Image

imagePath = 'spores'
outFolder = 'data'
for root, dirs, files in os.walk(imagePath):
    for file in files:
        filepath = os.path.join(root, file)
        image = Image.open(filepath)
        w, h = image.size
        cropped = image.crop((0, 0, w, int(h*0.9)))
        filepath_out = osp.join(outFolder, root.split('/')[-1])
        if os.path.exists(filepath_out) is False:
            os.makedirs(filepath_out)
        cropped.save(osp.join(filepath_out, file))
