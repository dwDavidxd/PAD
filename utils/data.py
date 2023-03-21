import numpy as np
import torch.utils.data as Data
from PIL import Image
import torch


class data_dataset(Data.Dataset):
    def __init__(self, img_path, clean_label_path, transform=None):
        self.transform = transform

        self.train_data = np.load(img_path)

        self.train_clean_labels = np.load(clean_label_path).astype(np.float32)
        self.train_clean_labels = torch.from_numpy(self.train_clean_labels).long()

    def __getitem__(self, index):
        img, clean_label = self.train_data[index], self.train_clean_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, clean_label

    def __len__(self):
        return len(self.train_data)


class data_dataset_with_ref(Data.Dataset):
    def __init__(self, img_path, clean_label_path, transform_1=None, transform_2=None, transform_3 =None):
        self.transform_1 = transform_1
        self.transform_2 = transform_2
        self.transform_3 = transform_3

        self.train_data = np.load(img_path)

        self.train_clean_labels = np.load(clean_label_path).astype(np.float32)
        self.train_clean_labels = torch.from_numpy(self.train_clean_labels).long()

    def __getitem__(self, index):
        img, clean_label = self.train_data[index], self.train_clean_labels[index]

        img = Image.fromarray(img)

        img = self.transform_1(img)

        img_ref = self.transform_3(img)

        img = self.transform_2(img)
        img = self.transform_3(img)

        return img, img_ref, clean_label

    def __len__(self):
        return len(self.train_data)
    

