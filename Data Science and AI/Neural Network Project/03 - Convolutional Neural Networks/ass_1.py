import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def preprocess_and_visualize(mat_filepath):
    # 1. Load the data
    # Note: Replace 'data' with the actual variable name inside your .mat file if different
    mat_contents = scipy.io.loadmat(mat_filepath)
    
    # Assuming data shape is (N, 16, 16, 3) or similar. Adjust axes if it's (16, 16, 3, N)
    rgb_patches = mat_contents['data'] 
    
    # Ensure shape is (N, 16, 16, 3) for the following operations
    if rgb_patches.shape[-1] != 3 and rgb_patches.shape[0] == 3:
        rgb_patches = np.transpose(rgb_patches, (3, 1, 2, 0)) # Example reshape if needed

    N = rgb_patches.shape[0]

    # 2. Convert to grayscale using the luminosity model
    # Y = 0.2126*R + 0.7152*G + 0.0722*B
    gray_patches = (0.2126 * rgb_patches[:, :, :, 0] + 
                    0.7152 * rgb_patches[:, :, :, 1] + 
                    0.0722 * rgb_patches[:, :, :, 2])
    
    # 3. Remove the mean pixel intensity of each image from itself
    # Calculate mean along the spatial dimensions (axis 1 and 2) and keep dims for broadcasting
    patch_means = np.mean(gray_patches, axis=(1, 2), keepdims=True)
    zero_mean_patches = gray_patches - patch_means
    
    # 4. Clip the data range at +/- 3 standard deviations (measured across ALL pixels)
    global_std = np.std(zero_mean_patches)
    clip_min = -3 * global_std
    clip_max = 3 * global_std
    clipped_patches = np.clip(zero_mean_patches, clip_min, clip_max)
    
    # 5. Map the +/- 3 std data range to [0.1, 0.9]
    # Formula: scaled = 0.1 + (val - min) * (0.9 - 0.1) / (max - min)
    normalized_patches = 0.1 + (clipped_patches - clip_min) * (0.9 - 0.1) / (clip_max - clip_min)
    
    # 6. Display 200 random sample patches
    num_samples_to_show = 200
    random_indices = np.random.choice(N, num_samples_to_show, replace=False)
    
    fig, axes = plt.subplots(20, 20, figsize=(12, 12))
    fig.suptitle('200 Random Original RGB Patches')
    for i, ax in enumerate(axes.flat):
        # Assuming original RGB data is in [0, 255], normalize to [0, 1] for matplotlib if needed
        img = rgb_patches[random_indices[i]]
        if img.max() > 1.0:
            img = img / 255.0
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(20, 20, figsize=(12, 12))
    fig.suptitle('200 Normalized Grayscale Patches')
    for i, ax in enumerate(axes.flat):
        ax.imshow(normalized_patches[random_indices[i]], cmap='gray', vmin=0.1, vmax=0.9)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    return normalized_patches

# Usage:
normalized_data = preprocess_and_visualize('assign3_data1.mat')