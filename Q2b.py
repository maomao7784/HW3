import cv2
import numpy as np
from scipy import ndimage

def laplacian_of_gaussian(image):
    # Apply Gaussian smoothing
    # cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) -> dst
    smoothed = cv2.GaussianBlur(image, (5,5), 1)

    # Apply Laplacian operator
    laplacian = cv2.Laplacian(smoothed, cv2.CV_64F)

    return laplacian

def main():
    images = ['/Users/linyinghsiao/Desktop/影像處理/HW3_updated/peppers.bmp', '/Users/linyinghsiao/Desktop/影像處理/HW3_updated/peppers_0.04.bmp']
    output_folder = "/Users/linyinghsiao/Desktop/影像處理/HW3_updated/Q2b/"

    for image_path in images:
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if image is loaded successfully
        if original is None:
            print(f"Error loading image {image_path}")
            continue

        log = laplacian_of_gaussian(original)

        # Normalize and convert to uint8
        log_norm = cv2.normalize(log, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        cv2.imwrite(output_folder + image_path.split('/')[-1], log_norm)

if __name__ == '__main__':
    main()
