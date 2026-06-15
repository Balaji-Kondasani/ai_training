import numpy as np

def resize_image_nearest_neighbor(image, new_height, new_width):
    """
    Resizes a 2D matrix (image) using Nearest Neighbor interpolation.
    """
    old_height, old_width = image.shape
    resized = np.zeros((new_height, new_width))
    
    for r in range(new_height):
        for c in range(new_width):
            old_r = int(np.floor(r * old_height / new_height))
            old_c = int(np.floor(c * old_width / new_width))
            resized[r, c] = image[old_r, old_c]
            
    return resized

def apply_box_blur(image):
    """
    Blurs an image using a standard 3x3 average filter.
    """
    kernel = np.ones((3, 3)) / 9.0
    img_h, img_w = image.shape
    output = np.zeros((img_h - 2, img_w - 2))
    
    for r in range(img_h - 2):
        for c in range(img_w - 2):
            slice_ = image[r : r + 3, c : c + 3]
            output[r, c] = np.sum(slice_ * kernel)
            
    return output

def main():
    print("=== Day 8: Computer Vision & Image Processing Basics ===")
    
    # 1. Mock Image representation (4x4 grayscale grid)
    mock_image = np.array([
        [10, 20, 30, 40],
        [15, 25, 35, 45],
        [20, 30, 40, 50],
        [25, 35, 45, 55]
    ], dtype=float)
    
    print("Original Grayscale Image (4x4 Matrix):")
    print(mock_image)
    print()
    
    # 2. Resizing Operation
    resized_image = resize_image_nearest_neighbor(mock_image, 6, 6)
    print("Resized Image (Nearest Neighbor, Upscaled to 6x6):")
    print(resized_image)
    print()
    
    # 3. Blurring Operation
    # Let's create a larger image to accommodate a 3x3 filter
    large_image = np.pad(mock_image, 1, mode='edge')
    blurred_image = apply_box_blur(large_image)
    print("Blurred Image (Average Box Filter):")
    print(blurred_image.round(2))
    print()
    
    # 4. OpenCV Concepts
    print("OpenCV (cv2) Workflow Concepts:")
    print("  * Image loading: cv2.imread('path', cv2.IMREAD_GRAYSCALE) -> returns a NumPy array.")
    print("  * Image resizing: cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR).")
    print("  * Image blurring: cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX).")
    print("  * Edge detection: cv2.Canny(image, low_threshold, high_threshold).")
    print("\nClassical CNN Architectures Summary:")
    print("  * LeNet-5 (1998): First successful CNN. Used for digit recognition (Conv -> Average Pool -> Dense).")
    print("  * AlexNet (2012): First deep CNN (8 layers, ReLU, Dropout, GPU training) winning ImageNet.")
    print("  * VGG-16 (2014): Modular design using small 3x3 conv filters stacked deeply.")
    print("  * ResNet (2015): Introduced residual skip connections to solve vanishing gradients in very deep networks (152+ layers).")

if __name__ == "__main__":
    main()
