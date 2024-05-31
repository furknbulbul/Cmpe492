import os
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import torch


class ImageDataset(Dataset):
    def __init__(self, root, phase="train", transform=None, target_transform=None):
        self.root = root
        self.phase = phase
        self.transform = transform
        self.target_transform = target_transform
        self.labels_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}

        if self.phase == "train":
            self.phase_root = os.path.join(root, 'train')
        elif self.phase == "test":
            self.phase_root = os.path.join(root, 'test')

    def __len__(self):
        lens = [len(os.listdir(os.path.join(self.phase_root, label))) for label in self.labels_map.keys()]
        self.lengths = lens
        return sum(lens)

    def __getitem__(self, idx):
        label = None
        for lbl, i in self.labels_map.items():
            label_path = os.path.join(self.phase_root, lbl)
            if idx < self.lengths[i]:
                label = lbl
                break
            idx -= self.lengths[i]
        img_name = os.listdir(label_path)[idx]
        img_path = os.path.join(label_path, img_name)
        image = read_image(img_path)
        image = image.to(torch.float32)
        image /= 255.0
        

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        

        return image, self.labels_map[label]
