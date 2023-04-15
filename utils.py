import torch
from PIL import Image
from torchvision.transforms import ToTensor

def pad_image(image, target_size, mode):
    width, height = image.size
    target_width, target_height = target_size
    padded_image = Image.new(mode, (target_width, target_height))
    p_x = (target_width - width) // 2
    p_y = (target_height - height) // 2
    padded_image.paste(image, (p_x, p_y))
    return padded_image


def my_collate_fn(batch):
    # Separate images, trimaps, and alpha mattings
    images, trimaps, alpha_mattings = zip(*batch)

    # Get the max width and height from the images
    max_width = max([i.size[0] for i in images])
    max_height = max([i.size[1] for i in images])
    target_size = (max_width, max_height)

    # Pad the images, trimaps, and alpha mattings to the max dimensions and convert them to tensors
    padded_images = []
    for image in images:
        padded_image = pad_image(image, target_size, 'RGB')
        tensor_image = ToTensor()(padded_image)
        padded_images.append(tensor_image)

    padded_trimaps = []
    for trimap in trimaps:
        padded_trimap = pad_image(trimap, target_size, 'L')
        tensor_trimap = ToTensor()(padded_trimap)
        padded_trimaps.append(tensor_trimap)

    padded_alpha_mattings = []
    for alpha_matting in alpha_mattings:
        padded_alpha_matting = pad_image(alpha_matting, target_size, 'L')
        tensor_alpha_matting = ToTensor()(padded_alpha_matting)
        padded_alpha_mattings.append(tensor_alpha_matting)

    # Stack the padded images, trimaps, and alpha mattings into batch tensors
    batch_images = torch.stack(padded_images)
    batch_trimaps = torch.stack(padded_trimaps)
    batch_alpha_mattings = torch.stack(padded_alpha_mattings)

    return batch_images, batch_trimaps, batch_alpha_mattings

