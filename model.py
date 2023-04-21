import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import models 
from torchsummary import summary
class Matting(nn.Module):
    def __init__(self):
        super(Matting, self).__init__()

        resnet = models.resnet18(pretrained=True)

        first_conv_layer = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        first_conv_layer.weight.data[:, :3, :, :] = resnet.conv1.weight.data
        first_conv_layer.weight.data[:, 3, :, :] = resnet.conv1.weight.data[:, 0, :, :]
        
        resnet.conv1 = first_conv_layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU(inplace=True)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=2)
        self.relu4 = nn.ReLU(inplace=True)
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def decoder(self, z, input_size):
        z = self.up1(z)
        z = self.conv1(z)
        z = self.relu1(z)
        z = self.up2(z)
        z = self.conv2(z)
        z = self.relu2(z)
        z = self.up3(z)
        z = self.conv3(z)
        z = self.relu3(z)
        z = self.up4(z)
        z = self.conv4(z)
        z = self.relu4(z)
        z = self.up5(z)
        z = self.conv5(z)
        z = self.relu5(z)
        z = self.conv6(z)
        z = self.sigmoid(z)

        # Add adaptive upsampling to match input size
        z = nn.Upsample(size=(input_size[2], input_size[3]), mode='bilinear', align_corners=True)(z)

        return z

    def forward(self,x):
        z = self.encoder(x)
        z = self.decoder(z, x.size())

        return z
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Matting().to(device)
    summary(model, (4, 320, 320))