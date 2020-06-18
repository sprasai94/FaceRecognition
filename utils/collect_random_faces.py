import os
import random
import shutil
path = "/home/sdevkota007/projects/resnet-face-pytorch/vgg-face-2/test/"

destination_folder = 'unknown'

i=0
for folder in os.listdir(path):
    image_folder = os.path.join(path, folder)
    images = os.listdir(image_folder)
    image = random.choice(images)
    image_path = os.path.join(image_folder, image)
    destination_path = os.path.join(destination_folder, image)
    shutil.copy(image_path, destination_path)

    if i==111:
        break