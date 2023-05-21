"""docstring"""
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import PIL.Image

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
        bbox = tuple(self.data.iloc[index, 1:5])
        if self.transform:
            image = self.transform(image)
        return image, label, bbox

transform_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(80, scale = (0.8, 1.0)),
    transforms.RandomApply([transforms.RandomAffine(degrees = 5, translate = (0.1, 0.1))], p = 0.5),
    transforms.ToTensor()
])

transform_simple = transforms.Compose([
    transforms.Resize(80),
    transforms.CenterCrop(80),
    transforms.ToTensor()
])

train_dataset = RSNADataset('/Users/taeyeonpaik/Downloads/rsna-pneumonia-detection-challenge/',
                           'stage_2_train_images_png/', 
                           'stage_2_train_labels.csv', 
                           transform = transform_augment)


test_dataset = RSNADataset('/Users/taeyeonpaik/Downloads/rsna-pneumonia-detection-challenge/',
                           'stage_2_test_images_png/', 
                           'stage_2_train_labels.csv', 
                           transform = transform_simple)

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False)
