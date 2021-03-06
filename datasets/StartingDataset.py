import torch
import os
from skimage import io
import pandas as pd
import matplotlib.pyplot as plt
import torchvision

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, images_dir, csv_file):
        self.csv_file = pd.read_csv(csv_file)
        self.images_dir = images_dir

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, index):

        image_id = os.path.join(self.images_dir,
                                self.csv_file.iloc[index, 0])
        image = io.imread(image_id)
        image = torch.Tensor(image)
        image = image.permute(2, 1, 0)
        labels = self.csv_file.iloc[index, 1]   
        #image.to(device)
        return image, labels

    def __showitem__(self, index):
        img, label = self.__getitem__(index)
        #plt.imshow(torchvision.transforms.ToPILImage()(img), interpolation="bicubic")
        image = img.int()
        plt.imshow(image)
        plt.show()
        #torchvision.transforms.ToPILImage()(img.permute(2, 0, 1)).show()
        #print(label)
