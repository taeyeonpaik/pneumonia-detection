"""docstring"""
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import PIL.Image
import numpy as np

class RSNADataset(Dataset):
    """docstring"""
    def __init__(self, root, image_dir, csv_file, transform = None):
        self.root = root
        self.image_dir = image_dir
        self.data = pd.read_csv(root + csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_name = self.data.iloc[index, 0]
        image_path = os.path.join(self.root, self.image_dir, image_name + '.png')
        image = PIL.Image.open(image_path).convert("RGB")
        label = self.data.iloc[index, -1]
        bbox = torch.Tensor(list(self.data.iloc[index, 1:5]))
        if self.transform:
            image = self.transform(image)
        return image, label, bbox

    def calculate_mean_std(self, image_folder):
        """docstring"""
        image_filenames = os.listdir(image_folder)
        pixel_sum, num_pixels = 0, 0
        sum_pixel_square = 0
        for filename in image_filenames:
            image_path = os.path.join(image_folder, filename)
            image = PIL.Image.open(image_path)
            image = np.array(image) / 255
            num_pixels += np.prod(image.shape)
            pixel_sum += np.sum(image)
            sum_pixel_square += np.sum(np.square(image))
        mean = pixel_sum / num_pixels
        std = np.sqrt(sum_pixel_square / num_pixels - np.square(mean))
        return mean, std

def get_data_loader():
    """docstring"""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(1024, scale = (0.8, 1.0)),
        transforms.RandomApply([transforms.RandomAffine((-5, 5), translate = (0.1, 0.1))], p = 0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.49011201641896834, 0.49011201641896834, 0.49011201641896834],
                                std = [0.2481732866714441, 0.2481732866714441, 0.2481732866714441])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.49011201641896834, 0.49011201641896834, 0.49011201641896834],
                                std = [0.2481732866714441, 0.2481732866714441, 0.2481732866714441])
    ])
    train_dataset = RSNADataset('/Users/taeyeonpaik/Downloads/rsna-pneumonia-detection-challenge/',
                            'stage_2_train_images_png/', 
                            'stage_2_train_labels.csv', 
                            transform = transform_train)
    test_dataset = RSNADataset('/Users/taeyeonpaik/Downloads/rsna-pneumonia-detection-challenge/',
                            'stage_2_test_images_png/', 
                            'stage_2_train_labels.csv', 
                            transform = transform_test)
    train_loader = DataLoader(train_dataset, batch_size = 4, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 4, shuffle = False)
    return train_loader, test_loader
