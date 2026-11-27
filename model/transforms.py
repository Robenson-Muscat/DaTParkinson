import random
from torch import eye, flip
import torch.nn.functional as F

# ============================================================
# 3. AUGMENTATION : FLIP + SCALING
# ============================================================

class RandomFlip3D:
    """Flip indépendant sur D, H et W."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        # image : (C, D, H, W)
        for axis in (1, 2, 3):
            if random.random() < self.p:
                image = flip(image, dims=(axis,))
        return image


class RandomScaling3D:
    """Zoom isotrope 3D par transformation affine."""

    def __init__(self, scale_range=0.15, p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, image):
        if random.random() >= self.p:
            return image

        scale = random.uniform(
            1.0 - self.scale_range,
            1.0 + self.scale_range,
        )

        # grid_sample 3D attend (N, C, D, H, W).
        batch = image.unsqueeze(0)

        theta = eye(3, 4, dtype=batch.dtype, device=batch.device).unsqueeze(0)

        # Une valeur > 1 dans theta réduit le champ visible (zoom-in).
        theta[0, 0, 0] = 1.0 / scale
        theta[0, 1, 1] = 1.0 / scale
        theta[0, 2, 2] = 1.0 / scale

        grid = F.affine_grid(
            theta,
            batch.shape,
            align_corners=False,
        )

        transformed = F.grid_sample(
            batch,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return transformed.squeeze(0)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for transform in self.transforms:
            image = transform(image)
        return image


def DaTVolumeTransform():
    return Compose([
        RandomFlip3D(p=0.5),
        RandomScaling3D(scale_range=0.15, p=0.5),
    ])