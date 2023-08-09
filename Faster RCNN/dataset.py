import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

class RSNADataset(Dataset):
    def __init__(self, dataframe, image_directory, transform = None):
        super().__init__()
        self.image_ids = dataframe['patientId']
        self.dataframe = dataframe
        self.image_directory = image_directory
        self.transform = transform
        
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        info = self.dataframe[self.dataframe['patientId'] == image_id]
        image = cv2.imread(f'{self.image_directory}/{image_id}.png', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        boxes, labels = info[['x', 'y', 'width', 'height']].values, info['Target'].values
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 3]
        target = {'boxes': boxes, 'labels': labels}
        if self.transform:
            sample = self.transform(**{'image': image, 'boxes': boxes, 'labels': labels})
            image, target['boxes'] = sample['image'], sample['boxes']
        return image, target, image_id

    def __len__(self):
        return self.image_ids.shape[0]
    
def get_train_transform():
    return A.Compose([A.Flip(0.5), A.Rotate(limit = 360), ToTensorV2(p = 1.0)], '''bbox_params = {'format': 'pascal_voc', 'label_fields': ['labels']})''')

def get_test_transform():
    return A.Compose([ToTensorV2(p = 1.0)], '''bbox_params = {'format': 'pascal_voc', 'label_fields': ['labels']}''')

def collate(batch):
    return tuple(zip(*batch))

def prepare_data():
    train_directory = f"/Users/taeyeonpaik/Downloads/rsna/train_images_png"
    test_directory = f"/Users/taeyeonpaik/Downloads/rsna/test_images_png"
    train_dataframe = pd.read_csv(f"/Users/taeyeonpaik/Downloads/rsna/stage_2_train_labels.csv")
    test_dataframe = pd.read_csv(f"/Users/taeyeonpaik/Downloads/rsna/stage_2_sample_submission.csv")
    train_dataframe_positive = pd.DataFrame(columns = ['patientId', 'x', 'y', 'width', 'height'])
    for i in range(len(train_dataframe)):
        if train_dataframe.loc[i]['Target'] == 1:
            train_dataframe_positive.append(train_dataframe.loc[i])
    train_dataset = RSNADataset(train_dataframe_positive, train_directory, get_train_transform())
    test_dataset = RSNADataset(test_dataframe, test_directory, get_test_transform())
    return train_dataset, test_dataset
    
def get_data_loader(batch_size):
    train_dataset, test_dataset = prepare_data()
    train_data_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate)
    test_data_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, collate_fn = collate)
    return train_data_loader, test_data_loader
