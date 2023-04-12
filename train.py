import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from model import Matting
import torch.nn as nn 
import torch.optim as optim

class MattingDataset(Dataset):
    def __init__(self, img_dir, trimap_dir, alpha_dir, transfrom=None):
        self.img_dir = img_dir
        self.trimap_dir = trimap_dir
        self.alpha_dir = alpha_dir
        self.image_files = os.listdir(img_dir)

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.image_files[index])
        trimap_path = os.path.join(self.trimap_dir, self.image_files[index])
        alpha_path = os.path.join(self.alpha_dir, self.image_files[index])

        image = Image.open(img_path).convert('RGB')
        trimap = Image.open(trimap_path).convert('L')
        alpha = Image.open(alpha_path).convert('L')

        return image, trimap, alpha

train_img_dir = None
train_trimap_dir = None
train_alpha_dir = None
eval_img_dir = None
eval_trimap_dir = None
eval_alpha_dir = None

train_dataset = MattingDataset(train_img_dir, train_trimap_dir, train_alpha_dir)
eval_dataset = MattingDataset(eval_img_dir, eval_trimap_dir, eval_alpha_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Matting().to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, trimaps, alphas) in enumerate(train_loader):
        images 