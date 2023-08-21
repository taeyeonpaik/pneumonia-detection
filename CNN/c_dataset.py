import os
import torch
import PIL.Image
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class RSNADataset(Dataset):
    def __init__(self, root, image_directory, data, transform = None):
        self.root = root
        self.image_directory = image_directory
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_name = self.data.iloc[index, 0]
        image_path = os.path.join(self.root, self.image_directory, image_name + '.png')
        image = PIL.Image.open(image_path).convert("RGB")
        label = torch.tensor(self.data.iloc[index, -1], dtype = torch.float32)
        bbox = torch.Tensor(list(self.data.iloc[index, 1:5]))
        if self.transform:
            image = self.transform(image)
        return image.to('cuda'), label.to('cuda'), bbox.to('cuda')

    def calculate_mean_std(self, image_folder):
        image_file_names = os.listdir(image_folder)
        pixel_sum, num_pixels = 0, 0
        sum_pixel_square = 0
        for file_name in image_file_names:
            image_path = os.path.join(image_folder, file_name)
            image = PIL.Image.open(image_path)
            image = np.array(image) / 255
            pixels_number += np.prod(image.shape)
            pixels_sum += np.sum(image)
            sum_pixel_square += np.sum(np.square(image))
        mean = pixels_sum / num_pixels
        standard_deviation = np.sqrt(sum_pixel_square / pixels_number - np.square(mean))
        return mean, standard_deviation

def get_data_loader():
    dataframe = pd.read_csv('/home/ec2-user/rsna/stage_2_train_labels.csv')
    train_data = dataframe[:-5000]
    valid_data = dataframe[-5000:-2500]
    test_data = dataframe[-2500:]
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(1024, scale = (0.8, 1.0)),
        transforms.RandomApply([transforms.RandomAffine((-5, 5), translate = (0.1, 0.1))], p = 0.5), transforms.ToTensor(),
        transforms.Normalize(mean = [0.49011201641896834, 0.49011201641896834, 0.49011201641896834], std = [0.2481732866714441, 0.2481732866714441, 0.2481732866714441])])
    transform_valid_test = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean = [0.49011201641896834, 0.49011201641896834, 0.49011201641896834], std = [0.2481732866714441, 0.2481732866714441, 0.2481732866714441])])
    train_dataset = RSNADataset('/home/ec2-user/rsna/', 'train_images_png/', train_data, transform = transform_train)
    valid_dataset = RSNADataset('/home/ec2-user/rsna/', 'train_images_png/', valid_data, transform = transform_valid_test)
    test_dataset = RSNADataset('/home/ec2-user/rsna/', 'train_images_png/', test_data, transform = transform_valid_test)
    train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    valid_dataloader = DataLoader(valid_dataset, batch_size = 1, shuffle = False)
    test_dataloader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
    return train_dataloader, valid_dataloader, test_dataloader
