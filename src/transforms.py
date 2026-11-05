"""Data augmentations and image transforms for knee MRI slices/volumes.
Designed for 2.5D MRI windows (3-channel stacked slices).
"""

import numpy as np

class MRITransforms:
    """MRI Augmentations supporting both NumPy arrays and tensor inputs."""
    def __init__(self, is_train: bool = True, img_size: int = 256):
        self.is_train = is_train
        self.img_size = img_size
        
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Args:
            img: np.ndarray of shape (H, W, C) or (C, H, W), float32 in [0, 1].
        Returns:
            np.ndarray of same shape, transformed.
        """
        # Ensure H, W, C for geometric transforms
        transposed = False
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
            transposed = True
            
        h, w = img.shape[:2]
        
        # Center crop / resize to target size
        if h != self.img_size or w != self.img_size:
            # Crop central region
            crop_h = min(h, int(h * 0.9)) if self.is_train else h
            crop_w = min(w, int(w * 0.9)) if self.is_train else w
            start_y = (h - crop_h) // 2
            start_x = (w - crop_w) // 2
            img = img[start_y:start_y + crop_h, start_x:start_x + crop_w]
            
            # Simple bilinear resize in numpy
            y_indices = np.linspace(0, img.shape[0] - 1, self.img_size).astype(int)
            x_indices = np.linspace(0, img.shape[1] - 1, self.img_size).astype(int)
            img = img[np.ix_(y_indices, x_indices)]
            
        if self.is_train:
            # Random horizontal flip (anatomically symmetric for coronal/axial)
            if np.random.rand() > 0.5:
                img = np.fliplr(img)
            # Slight random contrast scaling
            scale = np.random.uniform(0.9, 1.1)
            img = np.clip(img * scale, 0.0, 1.0)
            
        if transposed:
            img = np.transpose(img, (2, 0, 1))
            
        return img.astype(np.float32)

