import numpy as np
from dataloader.image_dataset import ImageDataset
class SiamaseDataset(ImageDataset):
    def __init__(self, root, phase="train", transform=None, target_transform=None):
        super(SiamaseDataset, self).__init__(root, phase, transform, target_transform)
        


    def __getitem__(self, idx):
     
        target = np.random.randint(0, 2)
        img1, label1 = super().__getitem__(idx)
        # target = 1 means the two images are from the same class
        # target = 0 means the two images are from different classes
        if target == 1:
            # Find another image from the same class
            while True:
                idx2 = np.random.randint(0, self.__len__())
                img2, label2 = super().__getitem__(idx2)
                if label1 == label2:
                    break
        else:
            # Find another image from a different class
            while True:
                idx2 = np.random.randint(0, self.__len__())
                img2, label2 = super().__getitem__(idx2)
                if label1 != label2:
                    break

        if  self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        
        return (img1, img2), target