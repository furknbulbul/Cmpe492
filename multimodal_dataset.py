import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer
import torch


class MultimodalDataset(Dataset):
    def __init__(self, root, phase="train", transform=None, target_transform=None):
        self.root = root
        self.phase = phase
        self.transform = transform
        self.target_transform = target_transform
        self.tokenize = get_tokenizer("basic_english")
        self.glove = GloVe(name='6B', dim=100)
        self.labels_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}

        if self.phase == "train":
            self.phase_root = os.path.join(root, 'train')
        elif self.phase == "test":
            self.phase_root = os.path.join(root, 'test')

        self.data = []

        for label in self.labels_map.keys():
            label_path = os.path.join(self.phase_root, label)
            for img_name in os.listdir(label_path):
                self.data.append((os.path.join(label_path, img_name), label))
                

    def __len__(self):
        lens = [len(os.listdir(os.path.join(self.phase_root, label))) for label in self.labels_map.keys()]
        self.lengths = lens
        return sum(lens)

    def __getitem__(self, idx):
        
        img_path, text = self.data[idx]
        label = self.labels_map[text]
        indices = torch.tensor([self.glove.stoi[word] for word in self.tokenize(text)])
        image = read_image(img_path).to(torch.float32)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, indices, label
