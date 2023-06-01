import cv2
import numpy as np
import pandas as pd
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as psnr
from tabulate import tabulate

def add_salt_pepper_noise(image, noise_ratio):
    noisy = random_noise(image, mode='s&p', salt_vs_pepper=0.5, amount=noise_ratio)
    noisy = np.array(255*noisy, dtype = 'uint8')
    return noisy

def mean_filter(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    filtered_image = cv2.filter2D(image, -1, kernel)
    filtered_image[(image == 0) | (image == 255)] = image[(image == 0) | (image == 255)]
    return filtered_image

def main():
    images = ['baboon.bmp', 'peppers.bmp']
    noise_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    table = []
    
    for image_path in images:
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        row = [image_path]
        for noise_ratio in noise_ratios:
            noisy = add_salt_pepper_noise(original, noise_ratio)
            filtered = mean_filter(noisy, 5)

            psnr_before = psnr(original, noisy)
            psnr_after = psnr(original, filtered)
            
            row.extend([f"{psnr_before:.2f}", f"{psnr_after:.2f}"])
            
        table.append(row)

    headers = ["Image", "10% Before", "10% After", "30% Before", "30% After",
               "50% Before", "50% After", "70% Before", "70% After", 
               "90% Before", "90% After"]

    df = pd.DataFrame(table, columns=headers)
    output_folder = "/Users/linyinghsiao/Desktop/影像處理/HW3_updated/Q1b/"
    df.to_csv(output_folder + 'psnr_table.csv', index=False)

if __name__ == '__main__':
    main()
