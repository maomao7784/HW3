import cv2
import numpy as np

def add_salt_pepper_noise(image, salt_prob, pepper_prob):
    '''
    Add salt and pepper noise to image
    salt_prob : Probability of the noise (range: 0-1)
    pepper_prob : Probability of the noise (range: 0-1)
    '''
    output = np.copy(image)

    # Salt mode
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in image.shape]
    output[tuple(coords)] = 255

    # Pepper mode
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper))
          for i in image.shape]
    output[tuple(coords)] = 0
    return output

output_folder = "/Users/linyinghsiao/Desktop/影像處理/HW3_updated/Q1a/"  # Define your output folder here

# Specify the image files
image_files = ['baboon.bmp', 'peppers.bmp']

# Specify the noise levels
noise_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

for image_file in image_files:
    image = cv2.imread(image_file, 0)  # Read the image in grayscale mode
    for noise_level in noise_levels:
        # Add the salt and pepper noise to the image
        noisy_image = add_salt_pepper_noise(image, noise_level, noise_level)

        # Save the noisy image
        cv2.imwrite(output_folder + f'{image_file[:-4]}_{int(noise_level*100)}perc_noise.bmp', noisy_image)
