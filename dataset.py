import cv2
import math
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

class RSNADataset(Dataset):
    def __init__(self, dataframe, image_directory, transform = None):
        super().__init__()
        self.image_ids = dataframe['patientId'] #.unique()
        self.dataframe = dataframe
        self.image_directory = image_directory
        self.transform = transform
        
    def __getitem__(self, index):
        info = self.dataframe.loc[index]
        image = cv2.imread(f'{self.image_directory}/{info[0]}.png', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        boxes, labels = info[1:5], info[5]
        # If bounding box is nan, replace with 0
        for i in range(len(boxes)):
            if math.isnan(boxes[i]):
                boxes[i] = 1
        boxes[2] += boxes[0]
        boxes[3] += boxes[1]
        sample = self.transform(**{'image': image, 'boxes': boxes})
        sample_boxes = torch.tensor(sample['boxes'].astype(np.float32))
        sample_labels = torch.tensor(labels, dtype = torch.int64)
        return sample['image'], {'boxes': sample_boxes, 'labels': sample_labels}, info[0]

    def __len__(self):
        return self.image_ids.shape[0]
    
def get_train_transform():
    return A.Compose([A.Flip(0.5), A.Rotate(limit = 15), ToTensorV2(p = 1.0)])

def get_valid_test_transform():
    return A.Compose([ToTensorV2(p = 1.0)])

def collate(batch):
    return tuple(zip(*batch))

def prepare_data():
    directory = f"/home/ec2-user/rsna/train_images_png" # "/Users/taeyeonpaik/Downloads/rsna/train_images_png"
    dataframe = pd.read_csv(f"/home/ec2-user/rsna/stage_2_train_labels.csv") # "/Users/taeyeonpaik/Downloads/rsna/stage_2_train_labels.csv"
    # Split images
    image_ids = dataframe['patientId']
    train_ids = image_ids[:-5000]
    valid_ids = image_ids[-5000:-2500]
    test_ids = image_ids[-2500:]
    train_dataframe = dataframe[dataframe['patientId'].isin(train_ids)]
    valid_dataframe = dataframe[dataframe['patientId'].isin(valid_ids)]
    test_dataframe = dataframe[dataframe['patientId'].isin(test_ids)]
    train_dataframe = train_dataframe.reset_index(drop = True)
    valid_dataframe = valid_dataframe.reset_index(drop = True)
    test_dataframe = test_dataframe.reset_index(drop = True)
    train_dataset = RSNADataset(train_dataframe, directory, get_train_transform())
    valid_dataset = RSNADataset(valid_dataframe, directory, get_valid_test_transform())
    test_dataset = RSNADataset(test_dataframe, directory, get_valid_test_transform())
    return train_dataset, valid_dataset, test_dataset
    
def get_data_loader(batch_size):
    train_dataset, valid_dataset, test_dataset = prepare_data()
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False, collate_fn = collate)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, collate_fn = collate)
    return train_dataloader, valid_dataloader, test_dataloader
