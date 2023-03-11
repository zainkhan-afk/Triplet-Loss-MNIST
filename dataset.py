from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import random
import torch
import glob
import cv2
import os 

class ASLDataset(Dataset):
    def __init__(self, all_paths, all_labels, img_size = (28, 28)):
        self.img_size = img_size
        self.all_paths = all_paths
        self.all_labels = all_labels

        self.all_paths = np.array(self.all_paths)
        self.all_labels = np.array(self.all_labels)
    
    def _load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, self.img_size)
        img = img.astype("float32")/255.0
        # img = img.transpose((2, 0, 1))
        img = img[np.newaxis, :, :]
        img = torch.from_numpy(img)

        return img

    def __len__(self):
        return len(self.all_paths)
    
    def __getitem__(self, idx):
        anchor_path = self.all_paths[idx]
        label = self.all_labels[idx]
        positive_paths = self.all_paths[(self.all_labels == label) & (self.all_paths != anchor_path)]
        negative_paths = self.all_paths[self.all_labels != label]


        positive_path = random.choice(positive_paths)
        negative_path = random.choice(negative_paths)

        anchor_img = self._load_image(anchor_path)
        positive_img = self._load_image(positive_path)      
        negative_img = self._load_image(negative_path)


        return anchor_img, positive_img, negative_img, torch.tensor([label])


if __name__ == "__main__":
    dl = ASLDataset("Dataset/ASL_DATASET/asl_alphabet_train/asl_alphabet_train")
    train_data_generator = DataLoader(dl, batch_size = 32, shuffle = True)
    train_data_iter = iter(train_data_generator)

    for anchor_imgs, positive_imgs, negative_imgs, actual_labels in train_data_iter:
        print(anchor_imgs.min(), anchor_imgs.max())
        exit()