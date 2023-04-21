import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from model import Matting
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torchvision 

from utils import pad_image, my_collate_fn

class MattingDataset(Dataset):
    def __init__(self, img_dir, trimap_dir, alpha_dir, transfrom=ToTensor()):
        self.img_dir = img_dir
        self.trimap_dir = trimap_dir
        self.alpha_dir = alpha_dir
        self.image_files = os.listdir(img_dir)
        self.trimap_files = os.listdir(trimap_dir)
        self.alpha_files = os.listdir(alpha_dir)
        self.transform = transfrom
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.image_files[index])
        trimap_path = os.path.join(self.trimap_dir, self.image_files[index])
        alpha_path = os.path.join(self.alpha_dir, self.image_files[index])

        image = Image.open(img_path).convert('RGB')
        trimap = Image.open(trimap_path).convert('L')
        alpha = Image.open(alpha_path).convert('L')
        #image_rgba = Image.open(img_path).convert('RGB')
        #image_rgba.putalpha(trimap)
        # transform1 = torchvision.transforms.Compose([
        #     torchvision.transforms.Grayscale(num_output_channels=3),
        #     torchvision.transforms.ToTensor()
        # ])
        # transform2 = torchvision.transforms.Compose([
        #     torchvision.transforms.Grayscale(num_output_channels=1),
        #     torchvision.transforms.ToTensor()
        # ])
        # transform3 = torchvision.transforms.Compose([
        #     torchvision.transforms.Grayscale(num_output_channels=1),
        #     torchvision.transforms.ToTensor()
        # ])

        # image = self.transform(image)
        # trimap = self.transform(trimap)
        # alpha = self.transform(alpha)
        
        #image_rgba = self.transform(image_rgba)
        return image, trimap, alpha
        #return image_rgba, alpha

def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss):
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'model': model,
             'optimizer': optimizer}
    torch.save(state, 'BEST_checkpoint.tar')


if __name__ == "__main__":
    train_img_dir = "Training_Dataset/Image/"
    train_trimap_dir = "Training_Dataset/Trimap/"
    train_alpha_dir = "Training_Dataset/Alpha/"
    eval_img_dir = "Evaluation_Dataset/Image/"
    eval_trimap_dir = "Evaluation_Dataset/Trimap/"
    eval_alpha_dir = "Evaluation_Dataset/Alpha/"

    train_dataset = MattingDataset(train_img_dir, train_trimap_dir, train_alpha_dir)
    eval_dataset = MattingDataset(eval_img_dir, eval_trimap_dir, eval_alpha_dir)

    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=my_collate_fn, shuffle=False, num_workers=2)
    eval_loader = DataLoader(eval_dataset, batch_size=8, collate_fn=my_collate_fn, shuffle=False, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Matting().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 100

    best_loss = float('inf')
    epochs_since_improvement = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        i = 0

        for images, trimaps, alphas in train_loader:
            i += 1
            print(i)
            #print(images.shape)
            #print(alphas.shape)
            images = images.to(device)
            #print(2)
            trimaps = trimaps.to(device)
            #print(3)
            alphas = alphas.to(device)
            #print(4)
        # for images_rgba, alphas in train_loader:
        #     print(images_rgba, alphas)
        #     images_rgba = images_rgba.to(device)
        #     alphas = alphas.to(device)

            optimizer.zero_grad()
            #print(5)

            #outputs = model(torch.cat([images, trimaps], dim=1))
            outputs = model(torch.cat((images, trimaps), dim=1))
            #print(6)
            #outputs = model(images_rgba)
            print(outputs.shape)
            print(alphas.shape)
            loss = criterion(outputs, alphas)
            #print(7)
            loss.backward()
            #print(8)
            optimizer.step()
            #print(9)
            print(loss.item())
            running_loss+=loss.item()
        epoch_loss =running_loss/len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}")
        
        model.eval()
        eval_loss=0.0
        for images, trimaps, alphas in eval_loader:
            images_eval = images.to(device)
            trimaps_eval = trimaps.to(device)
            alphas_eval = alphas.to(device)

            optimizer.zero_grad()

            outputs_eval = model(torch.cat((images_eval, trimaps_eval), dim=1))
            loss_eval = criterion(outputs_eval, alphas_eval)
            loss_eval.backward()

            optimizer.step()
            eval_loss+=loss_eval.item()
        epoch_loss_eval =eval_loss/len(eval_loader)

        ifbest = epoch_loss_eval < best_loss

        best_loss = min(epoch_loss_eval, best_loss)
        if not ifbest:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
            save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss)

        print(f"Epoch eval {epoch+1}/{num_epochs} - Loss: {epoch_loss_eval:.4f}")

