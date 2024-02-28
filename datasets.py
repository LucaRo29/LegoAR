import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets.folder import default_loader

from torchvision.datasets.folder import pil_loader
from torchvision.datasets.vision import VisionDataset
from pathlib import Path
import os
import re
from PIL import Image, ImageEnhance
import configparser
import cv2
import sys
import numpy as np 

config = configparser.ConfigParser()
config.read('config.ini')

TRAIN_DIR = config.get('DATA','TRAIN_DIR')
VAL_DIR = config.get('DATA','VAL_DIR')

#DATA_DIR = config.get('DATA','DATA_DIR')
VAL_SPLIT = config.getfloat('DATA','VAL_SPLIT')
TEST_SPLIT = config.getfloat('DATA','TEST_SPLIT')

IMAGE_SIZE = config.getint('DATA','IMAGE_SIZE')


BATCH_SIZE = config.getint('TRAINING','batch_size')
NUM_WORKERS = config.getint('TRAINING','num_workers')


class LegoARDataset(torch.utils.data.Dataset):
    def __init__(self, root, classes=None, 
                 transforms=transforms.Compose([transforms.ToTensor()])):
        self.root = root
        self.transforms = transforms
        if classes is None:
            self.classes = [cls for cls in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, cls))]
            self.classes.sort(key= lambda x: int(x.split("_")[1]))
            
        else:
            self.classes=classes
        
        self.class_to_idx = {}#{cls_name: i for i, cls_name in enumerate(self.classes)}

        for cls in self.classes:
            key = cls
            value = int(cls.split('_')[1])
            self.class_to_idx[key] = value

        self.files = []
        self.labels = []
        for cls in self.classes:
            clsfiles = os.listdir(os.path.join(self.root, cls))
            self.files.extend(clsfiles)
            self.labels.extend([self.class_to_idx.get(cls)]*len(clsfiles))        
        assert len(self.files)==len(self.labels)
       
        print(f'found {len(self.classes)} classes and {len(self.files)} files in f{root}')

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        image_path = os.path.join(self.root, f"step_{self.labels[idx]}", self.files[idx])
        image = pil_loader(image_path)

        if self.transforms is not None:
            image = self.transforms(image)

        return (image, self.labels[idx])
    
    def tensor2image(self, t: torch.Tensor):
        return (t.detach().cpu().numpy().transpose(
            (1, 2, 0)) * 255).astype(np.uint8)





def get_train_transform():
    train_transform = transforms.Compose([
        transforms.Lambda(augment),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.AutoAugment(),
        transforms.ToTensor(),
        transforms.Normalize( mean= [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
    ])
    return train_transform


# Validation transforms
def get_val_transform():
    val_transform = transforms.Compose([       
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize( mean= [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
    ])
    return val_transform

def get_real_transform():
    real_transform = transforms.Compose([
        
        BrightnessTransform(brightness_factor=2),
        ContrastTransform(contrast_factor=1.25),
        SquarePad(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize( mean= [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
    ])
    return real_transform

    

def get_padding(obj):

    min_size = np.max(obj.shape) +50
  
    size = torch.randint(low = min_size, high = int(min_size*1.5),size =(1,1)).squeeze().item()

    sidetoside_margin = size-obj.shape[1]
    right_pad = torch.randint(low= int(sidetoside_margin*0.25 ),high = int(sidetoside_margin*0.75 ),size =(1,1)).squeeze().item()
    left_pad = size-obj.shape[1]-right_pad


    toptobottom_margin = size-obj.shape[0]
    top_pad = torch.randint(low= int(toptobottom_margin*0.25 ),high = int(toptobottom_margin*0.75 ),size =(1,1)).squeeze().item()

    bottom_pad = size-obj.shape[0]-top_pad

    return [(top_pad,bottom_pad),(left_pad,right_pad)]


def augment(img ):
    img = np.array(img)
    
    Y,X = np.where(~cv2.inRange(img, (255,255,255),(255,255,255)))
    bbox = (Y.min(), Y.max(), X.min(), X.max())
    b, t, l, r = bbox

    obj = img[b:t, l:r]

    padding = get_padding(obj)
   
    img =  np.pad(obj, ( padding[0],padding[1], (0,0)), 
                  mode='constant', constant_values=255) 
    
    return Image.fromarray(img)


def get_full_dataset():
    dataset = LegoARDataset(
        TRAIN_DIR ,
        transforms=get_train_transform(IMAGE_SIZE, pretrained=True))        

    return dataset


def get_real_test_set(Path,classes = None):

    dataset = LegoARDataset(
                            Path,
                            classes,
                            transforms=(get_real_transform())
                            )

    loader = DataLoader(
                        dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS
                        )

    # print('\nLabel mapping:')
    # print(dataset.class_to_idx )

    return dataset, dataset.classes, loader


def get_datasets(classes = None):

    # torch.manual_seed(0)
    """
    Function to prepare the Datasets.
    :param pretrained: Boolean, True or False.
    Returns the training and valation datasets along
    with the class names.
    """

    dataset = LegoARDataset(
        TRAIN_DIR,
        classes,
        transforms=(get_train_transform())
    )

    full_dataset_val  = LegoARDataset(
        TRAIN_DIR,
        classes,
        transforms=(get_val_transform())
    )

    
    assert len(VAL_DIR) != VAL_SPLIT # len(VAL_DIR) cannot be 0 if VAL_SPLIT is 0

    if (VAL_SPLIT > 0 and TEST_SPLIT > 0):
        dataset_size = len(dataset)
        # Calculate the validation and test dataset sizes.
        val_size = int(VAL_SPLIT * dataset_size)
        test_size = int(TEST_SPLIT * dataset_size)
        # Randomize the data indices.
        np.random.seed(42)
        indices = np.random.permutation(dataset_size)
        print(indices[:50])

        file_name = 'indices.txt'
        with open(file_name, 'w') as file:
            # Write the formatted string to the file
            file.write(np.array2string(indices, separator=','))

        # Training, validation, and test sets.
        dataset_train = Subset(dataset, indices[:-val_size-test_size])
        dataset_val = Subset(full_dataset_val, indices[-val_size-test_size:-test_size])
        dataset_test = Subset(full_dataset_val, indices[-test_size:])

    else:
        dataset_val = LegoARDataset(
        VAL_DIR,
        transforms=(get_val_transform(IMAGE_SIZE, pretrained))
        )

        dataset_train = dataset
        

    # print('\nLabel mapping:')
    # print(dataset.class_to_idx )

    return dataset_train, dataset_val, dataset_test, dataset.classes


def get_eval_datasets():
    

    dataset = LegoARDataset(
        TRAIN_DIR,
        transforms=(get_val_transform())
    )

    
    assert len(VAL_DIR) != VAL_SPLIT # len(VAL_DIR) cannot be 0 if VAL_SPLIT is 0

    if (VAL_SPLIT > 0 and TEST_SPLIT > 0):
        dataset_size = len(dataset)
        # Calculate the validation and test dataset sizes.
        val_size = int(VAL_SPLIT * dataset_size)
        test_size = int(TEST_SPLIT * dataset_size)
        # Randomize the data indices.
        indices = torch.randperm(dataset_size).tolist()
        #print(indices[:50])
        # Training, validation, and test sets.
        dataset_train = Subset(dataset, indices[:-val_size-test_size])
        dataset_val = Subset(dataset, indices[-val_size-test_size:-test_size])
        dataset_test = Subset(dataset, indices[-test_size:])

    else:
        dataset_val = LegoARDataset(
        VAL_DIR,
        transforms=(get_val_transform())
        )

        dataset_train = dataset
        


    #print('\nLabel mapping:')
    #print(dataset.class_to_idx )

    return dataset_train, dataset_val, dataset_test, dataset.classes

def get_data_loaders(dataset_train, dataset_val, dataset_test):
    """
    Prepares the training and valation data loaders.
    :param dataset_train: The training dataset.
    :param dataset_val: The valation dataset.
    Returns the training and valation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        dataset_val, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        dataset_test, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )
    return train_loader, val_loader, test_loader


class ResizeAndPadTransform:
    def __init__(self, size=1024, fill_color=(255, 255, 255)):
        self.size = size
        self.fill_color = fill_color

    def __call__(self, img):
        # Resize the image while maintaining the aspect ratio
        img.thumbnail((self.size, self.size), Image.ANTIALIAS)

        # Create a new blank image with a square size
        new_img = Image.new("RGB", (self.size, self.size), self.fill_color)

        # Calculate the coordinates to paste the resized image
        x_offset = (self.size - img.width) // 2
        y_offset = (self.size - img.height) // 2

        # Paste the resized image onto the new blank image
        new_img.paste(img, (x_offset, y_offset))

        # Convert PIL image to PyTorch tensor
        tensor_img = transforms.ToTensor()(new_img)

        return tensor_img

class BrightnessTransform:
    def __init__(self, brightness_factor):
        self.brightness_factor = brightness_factor

    def __call__(self, img):
        # Convert PIL image to PIL ImageEnhance object
        enhancer = ImageEnhance.Brightness(img)

        # Adjust the brightness using the factor
        img = enhancer.enhance(self.brightness_factor)

        return img

class SquarePad(object):
    def __call__(self, img):
        width, height = img.size
        max_dim = max(width, height)
        padded_image = Image.new('RGB', (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(img, ((max_dim - width) // 2, (max_dim - height) // 2))

        padded_image = padded_image.transpose(Image.FLIP_TOP_BOTTOM)

        #padded_image = padded_image.crop((50, 50, width - 50,height- 50))

        return padded_image


class ContrastTransform:
    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def __call__(self, img):
        # Convert PIL image to PIL ImageEnhance object
        enhancer = ImageEnhance.Contrast(img)

        # Adjust the contrast using the factor
        img = enhancer.enhance(self.contrast_factor)

        return img



    
# Mean:  [243.03871839 ,243.92204199 ,246.37171143]
# Std:  [38.96630051, 36.04916466 ,32.80039724]

