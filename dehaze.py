import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def dehaze_image(image_path):
    print(image_path)
    data_hazy = Image.open(image_path)
    data_hazy_np = np.asarray(data_hazy)/255.0

    data_hazy = torch.from_numpy(data_hazy_np).float()
    data_hazy = data_hazy.permute(2,0,1)
    device = torch.device('cpu')
    data_hazy = data_hazy.to(device)

    data_hazy = data_hazy.unsqueeze(0)
    dehaze_net = net.dehaze_net().to(device)
    dehaze_net.load_state_dict(torch.load('snapshots/dehazer.pth', map_location=torch.device('cpu')))

    clean_image = dehaze_net(data_hazy)

    clean_image_np = clean_image.detach().cpu().numpy()[0].transpose(1,2,0)

    save_dir = "results/"
    os.makedirs('results/test_images', exist_ok=True)

    psnr_score = psnr(data_hazy_np, clean_image_np, data_range=1.0)
    ssim_score = ssim(data_hazy_np, clean_image_np, multichannel=True, data_range=1.0, win_size=3)
    print(f"PSNR: {psnr_score:.4f}, SSIM: {ssim_score:.4f}")
    torchvision.utils.save_image(torch.cat((torch.from_numpy(data_hazy_np.transpose(2,0,1)).unsqueeze(0), clean_image),0), save_dir + image_path.split("/")[-1])


if __name__ == '__main__':

    test_list = glob.glob("test_images/*")

    for image in test_list:
        dehaze_image(image)
        print(image, "done!")
