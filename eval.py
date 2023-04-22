import numpy as np
import torch
from model import Matting
from PIL import Image
import torchvision
if __name__=="__main__":
    ckpt = torch.load('BEST_checkpoint.tar')
    loaded_model = ckpt['model'].cpu()
    img_path = 'Evaluation_Dataset/Image/0.png'
    trimap_path = 'Evaluation_Dataset/Trimap/0.png'
    image = Image.open(img_path).convert('RGB')
    trimap = Image.open(trimap_path).convert('L')
    image = torchvision.transforms.ToTensor()(image)
    trimap = torchvision.transforms.ToTensor()(trimap)
    print(image.shape)
    print(trimap.shape)
    X_test = torch.cat((image, trimap), dim=0)
    X_test = X_test.reshape((1,4,X_test.shape[1],X_test.shape[2]))
    input_size = X_test.size()
    predicted_result = loaded_model.forward(X_test)
    predicted_result = predicted_result.detach()
    if predicted_result.is_cuda:
        predicted_result = predicted_result.cpu()

    #convert tensor to image
    predicted_result = predicted_result.numpy()
    predicted_result = predicted_result.reshape((input_size[2], input_size[3]))
    predicted_result = predicted_result
    predicted_result = predicted_result.astype(np.uint8)
    predicted_result = Image.fromarray(predicted_result)
    alpha_path  = 'Evaluation_Dataset/Alpha/0.png'
    alpha = Image.open(alpha_path).convert('L')
    # Change Alpha to numpy array
    alpha = np.array(alpha)
    # calculate the difference between the predicted result and the ground truth using the L1 norm
    diff = np.sum(np.abs(predicted_result - alpha))
    print(diff)
    predicted_result.save('predicted_result.png')