import os
import random
import shutil
from torch.utils.data import Dataset, random_split
# Set the path to the folder containing the images
data_dir = '/home/vince/datasets/nuscenes_kitti_exported'


class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.img_files = os.listdir(os.path.join(data_dir,"image_2"))
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        # Return the image as a tensor, e.g. using torchvision.transforms



# Create a CustomDataset object
dataset = CustomDataset(data_dir)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
# os.removedirs( os.path.join(data_dir, 'training'))
# os.removedirs( os.path.join(data_dir, 'testing'))
# os.removedirs( os.path.join(data_dir, 'ImageSets'))
# Create directories for training and testing sets
os.makedirs( os.path.join(data_dir, 'training','image_2'), exist_ok=True)
os.makedirs( os.path.join(data_dir, 'training','label_2'), exist_ok=True)
os.makedirs( os.path.join(data_dir, 'training','calib'), exist_ok=True)
os.makedirs( os.path.join(data_dir, 'testing','image_2'), exist_ok=True)
os.makedirs( os.path.join(data_dir, 'testing','label_2'), exist_ok=True)
os.makedirs( os.path.join(data_dir, 'testing','label_2'), exist_ok=True)
os.makedirs( os.path.join(data_dir, 'testing','calib'), exist_ok=True)
os.makedirs( os.path.join(data_dir,'ImageSets'), exist_ok=True)
# Copy the files from the original folder to the training and testing directories
train = []
test = []
for i, dataset in enumerate([train_dataset, test_dataset]):
    if i == 0:
        dst_dir = os.path.join(data_dir,'training')
    else:
        dst_dir = os.path.join(data_dir,'testing')
    for j in range(len(dataset)):
        img_name = dataset.dataset.img_files[dataset.indices[j]].split('.')[0]
        shutil.copy( os.path.join(data_dir, "image_2",img_name+".png") , os.path.join(dst_dir, "image_2",img_name+".png"))
        shutil.copy( os.path.join(data_dir, "label_2",img_name+".txt") , os.path.join(dst_dir, "label_2",img_name+".txt"))
        shutil.copy( os.path.join(data_dir, "calib",img_name+".txt") , os.path.join(dst_dir, "calib",img_name+".txt"))
        if i==0:
            train.append(img_name)
        else:
            test.append(img_name)
split_size= int(0.2 * len(train))
train_valid,valid,train=random_split(train , [split_size, split_size,len(train)-split_size-split_size])

with open(os.path.join(data_dir,"ImageSets/test.txt"), 'w') as f:
    f.writelines("%s\n" % line for line in test)
with open(os.path.join(data_dir,"ImageSets/train.txt"), 'w') as f:
    f.writelines("%s\n" % line for line in train)
with open(os.path.join(data_dir,"ImageSets/val.txt"), 'w') as f:
    f.writelines("%s\n" % line for line in valid)
with open(os.path.join(data_dir,"ImageSets/trainval.txt"), 'w') as f:
    f.writelines("%s\n" % line for line in train_valid)