import numpy as np

def convolution2d(image, kernel, stride=1, padding=0):
    """
    Computes a manual 2D convolution of an image with a single kernel channel.
    """
    # Apply padding to the image bounds if requested
    if padding > 0:
        image = np.pad(image, padding, mode='constant', constant_values=0)
        
    img_h, img_w = image.shape
    kernel_h, kernel_w = kernel.shape
    
    # Calculate output dimensions
    out_h = int((img_h - kernel_h) / stride) + 1
    out_w = int((img_w - kernel_w) / stride) + 1
    
    output = np.zeros((out_h, out_w))
    
    # Perform element-wise sliding multiplication
    for r in range(out_h):
        for c in range(out_w):
            r_start = r * stride
            r_end = r_start + kernel_h
            c_start = c * stride
            c_end = c_start + kernel_w
            
            image_slice = image[r_start:r_end, c_start:c_end]
            output[r, c] = np.sum(image_slice * kernel)
            
    return output

def max_pooling2d(matrix, pool_size=2, stride=2):
    """
    Computes a manual 2D MaxPooling over a matrix.
    """
    mat_h, mat_w = matrix.shape
    
    # Calculate output dimensions
    out_h = int((mat_h - pool_size) / stride) + 1
    out_w = int((mat_w - pool_size) / stride) + 1
    
    output = np.zeros((out_h, out_w))
    
    for r in range(out_h):
        for c in range(out_w):
            r_start = r * stride
            r_end = r_start + pool_size
            c_start = c * stride
            c_end = c_start + pool_size
            
            matrix_slice = matrix[r_start:r_end, c_start:c_end]
            output[r, c] = np.max(matrix_slice)
            
    return output

def main():
    print("=== Day 8: CNN & Computer Vision Foundations (Manual Operations) ===")
    
    # 1. Input Image representation (6x6 matrix, e.g., representing a vertical edge)
    image = np.array([
        [10, 10, 10,  0,  0,  0],
        [10, 10, 10,  0,  0,  0],
        [10, 10, 10,  0,  0,  0],
        [10, 10, 10,  0,  0,  0],
        [10, 10, 10,  0,  0,  0],
        [10, 10, 10,  0,  0,  0]
    ], dtype=float)
    
    # 2. Convolution Kernel: Sobel Vertical Edge Detector Filter (3x3)
    vertical_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=float)
    
    print("Input Image (6x6):")
    print(image)
    print("\nVertical Edge Detection Kernel (3x3):")
    print(vertical_kernel)
    
    # 3. Apply Convolution (stride=1, padding=0)
    conv_output = convolution2d(image, vertical_kernel, stride=1, padding=0)
    print(f"\nConvolution Result (4x4, stride=1):")
    print(conv_output)
    
    # 4. Apply MaxPooling (pool_size=2, stride=2)
    pooled_output = max_pooling2d(conv_output, pool_size=2, stride=2)
    print(f"\nMaxPooling Result (2x2, pool_size=2, stride=2):")
    print(pooled_output)
    
    print("\nSummary of Concepts:")
    print("  * Convolution: Extracts local features (like edges, corners) by sliding kernels across the image.")
    print("  * MaxPooling: Redundancy reduction and downsampling, preserving only the strongest feature activations while making the model translation-invariant.")

if __name__ == "__main__":
    main()
