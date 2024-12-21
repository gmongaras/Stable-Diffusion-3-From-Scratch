import torch
import torch.nn.functional as F

def patchify(images, patch_size):
    """
    Patchify a batch of images using a sliding window approach with padding and flatten the patches.
    
    Args:
        images: Tensor of shape (N, C, H, W), where N is batch size, C is number of channels, H and W are height and width of the image.
        patch_size: Size of each patch (patch_height, patch_width).

    Returns:
        patches: Tensor of shape (N, num_patches, patch_size * patch_size * C), containing flattened image patches.
    """
    N, C, H, W = images.shape
    patch_height, patch_width = patch_size
    
    # Calculate necessary padding to make the image dimensions multiples of the patch size
    pad_h = (patch_height - H % patch_height) % patch_height
    pad_w = (patch_width - W % patch_width) % patch_width

    # Pad the image
    images_padded = F.pad(images, (0, pad_w, 0, pad_h), mode='constant', value=0)
    
    # Calculate the new height and width after padding
    H_padded, W_padded = images_padded.shape[2], images_padded.shape[3]

    # Unfold (patchify) the image using sliding windows
    patches = images_padded.unfold(2, patch_height, patch_height).unfold(3, patch_width, patch_width)
    
    # Reshape to (N, num_patches, C, patch_height, patch_width)
    patches = patches.contiguous().view(N, C, -1, patch_height, patch_width).permute(0, 2, 1, 3, 4)
    
    # Flatten the last 3 dimensions (C, patch_height, patch_width) to (patch_size * patch_size * C)
    patches = patches.contiguous().view(N, -1, patch_height * patch_width * C)

    return patches



def unpatchify(patches, patch_size, original_shape):
    """
    Reconstruct the original images from flattened patches.
    
    Args:
        patches: Tensor of shape (N, num_patches, patch_size * patch_size * C), containing flattened patches.
        patch_size: Size of each patch (patch_height, patch_width).
        original_shape: Tuple of the original unpadded image shape (H_original, W_original).
        
    Returns:
        images: Tensor of shape (N, C, H_original, W_original), the reconstructed images.
    """
    N, num_patches, patch_dim = patches.shape
    patch_height, patch_width = patch_size
    H_original, W_original = original_shape
    
    # Compute the number of patches along height and width
    num_patches_h = (H_original + patch_height - 1) // patch_height
    num_patches_w = (W_original + patch_width - 1) // patch_width
    
    # Reshape patches back to (N, num_patches_h, num_patches_w, C, patch_height, patch_width)
    C = patch_dim // (patch_height * patch_width)
    patches = patches.view(N, num_patches_h, num_patches_w, C, patch_height, patch_width)
    
    # Permute and reshape to (N, C, H_padded, W_padded)
    images_padded = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    images_padded = images_padded.view(N, C, num_patches_h * patch_height, num_patches_w * patch_width)
    
    # Crop the images back to the original unpadded size
    images = images_padded[:, :, :H_original, :W_original]
    
    return images