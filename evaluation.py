from skimage import metrics
from piq import fsim
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms.functional as Fe
from pytorch_msssim import ms_ssim


def compute_ssim(img1,img2):
    img1 = img1.convert('L')
    img2 = img2.convert('L')
    # Convert images to arrays
    arr1 = np.array(img1)
    arr2 = np.array(img2)

    # Calculate SSIM
    ssim = metrics.structural_similarity(arr1, arr2, multichannel=True)

    return(ssim)
def compute_ms_ssim(img1,img2):
    # Resize both images to the same size
    size = (256, 256)
    image1 = Fe.resize(img1, size)
    image2 = Fe.resize(img2, size)

    # Convert the PIL images to PyTorch tensors
    tensor1 = Fe.to_tensor(image1)
    tensor2 = Fe.to_tensor(image2)

    # Calculate MS-SSIM
    ms_ssim_value = ms_ssim(tensor1.unsqueeze(0), tensor2.unsqueeze(0), data_range=1.0, size_average=True)

    return ms_ssim_value.item()
def preprocess(img):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor
def compute_p_index(img1,img2):
    model = models.vgg16(pretrained=True)
    model.eval()
    img_tensor1 = preprocess(img1)
    img_tensor2 = preprocess(img2)
    with torch.no_grad():
        feat1 = model(img_tensor1)
        feat2 = model(img_tensor2)
    feat1 = F.normalize(feat1, p=2, dim=1)
    feat2 = F.normalize(feat2, p=2, dim=1)
    diff = feat1 - feat2
    return (torch.norm(diff, p=2)).item()

def compute_fsim(img1,img2):
    # Convert the images to tensors
    img1_tensor = torch.tensor(np.array(img1)).permute(2, 0, 1).float().unsqueeze(0) / 255.
    img2_tensor = torch.tensor(np.array(img2)).permute(2, 0, 1).float().unsqueeze(0) / 255.

    # Compute the FSIM
    fsim_value = fsim(img1_tensor, img2_tensor, data_range=1.0, reduction='mean')

    # Print the FSIM value
    return(fsim_value.item())

def compute(img1,img2):
    ssim = compute_ssim(img1,img2)
    ms_ssim = compute_ms_ssim(img1,img2)
    p_index = compute_p_index(img1,img2)
    fsim = compute_fsim(img1,img2)
    print(ssim,ms_ssim,p_index,fsim)