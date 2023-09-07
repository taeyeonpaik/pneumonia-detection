import cv2
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

        # I need to make it so that it doesn't only check for when the patient id equals the image id. I need to go 
        # one by one to make sure that they are all separate. Otherwise, they're gonna be a 2d info df
        # image_id = self.image_ids[index]
        # info = self.dataframe[self.dataframe['patientId'] == image_id]
        info = self.dataframe.loc[index]
        image = cv2.imread(f'{self.image_directory}/{info[0]}.png', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        boxes, labels = info[1:5], info[5]
        boxes[2] += boxes[0]
        boxes[3] += boxes[1]
        sample = self.transform(**{'image': image, 'boxes': boxes})
        target = {'boxes': torch.tensor(sample['boxes'].astype(np.float32)), 'labels': torch.tensor(labels.astype(np.int64))}
        return sample['image'], target, info[0]

    def __len__(self):
        return self.image_ids.shape[0]
    
def get_train_transform():
    return A.Compose([A.Flip(0.5), A.Rotate(limit = 15), ToTensorV2(p = 1.0)])

def get_valid_test_transform():
    return A.Compose([ToTensorV2(p = 1.0)])

def collate(batch):
    return tuple(zip(*batch))

def prepare_data():
    directory = f"/home/ec2-user/rsna/train_images_png" #"/Users/taeyeonpaik/Downloads/rsna/train_images_png"
    dataframe = pd.read_csv(f"/home/ec2-user/rsna/stage_2_train_labels.csv") #"/Users/taeyeonpaik/Downloads/rsna/stage_2_train_labels.csv"
    # dataframe_positive = pd.DataFrame(columns = ['patientId', 'x', 'y', 'width', 'height', 'Target'])
    # k = 0
    # for i in range(len(dataframe)):
    #     if dataframe.loc[i]['Target'] == 1:
    #         dataframe_positive.loc[k] = dataframe.loc[i]
    #         k += 1
    image_ids = dataframe['patientId'] #.unique()
    train_ids = image_ids[:-5000]
    valid_ids = image_ids[-5000:-2500]
    test_ids = image_ids[-2500:]
    train_dataframe = dataframe[dataframe['patientId'].isin(train_ids)]
    valid_dataframe = dataframe[dataframe['patientId'].isin(valid_ids)]
    test_dataframe = dataframe[dataframe['patientId'].isin(test_ids)]
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
