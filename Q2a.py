import cv2
import numpy as np

def sobel_edge_detection(image):
    # Define Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)

    # Convolve the image with the Sobel kernels
    g_x = cv2.filter2D(image, -1, sobel_x)
    g_y = cv2.filter2D(image, -1, sobel_y)

    # Calculate the magnitude of gradients
    g = np.hypot(g_x, g_y)
    g = g / g.max() * 255
    return g.astype('uint8')

def main():
    images = ['/Users/linyinghsiao/Desktop/影像處理/HW3_updated/peppers.bmp', '/Users/linyinghsiao/Desktop/影像處理/HW3_updated/peppers_0.04.bmp']
    output_folder = "/Users/linyinghsiao/Desktop/影像處理/HW3_updated/Q2a/"

    for image_path in images:
        original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        edge_map = sobel_edge_detection(original)

        # Save the edge map to the output folder
        filename = image_path.split('/')[-1]  # Get the filename from the path
        cv2.imwrite(output_folder + 'edge_map_' + filename, edge_map)

if __name__ == '__main__':
    main()
