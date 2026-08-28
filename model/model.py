from pathlib import Path

import numpy as np

from monai.networks.nets import DenseNet121
import torch

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from monai.networks.nets import DenseNet121
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    Spacing,
    NormalizeIntensity,
    CropForeground,
    ResizeWithPadOrCrop,
)


TARGET_SPACING = (2.46, 2.46, 2.46)
TARGET_SIZE = (112, 112, 80)





class DATScanModel:
    """
    Model wrapper for Parkinson DAT-scan classification (DenseNet121 3D architecture).

    """

    def __init__(
        self,
        checkpoint_paths: Iterable[Path],
        device: str | None = None,
    ) -> None:

        self.checkpoint_paths = [
            Path(path) for path in checkpoint_paths
        ]

        if len(self.checkpoint_paths) == 0:
            raise ValueError(
                "No checkpoint was provided."
            )

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        self.models = []

        for checkpoint_path in self.checkpoint_paths:

            model = self._create_model()

            self._load_checkpoint(
                model,
                checkpoint_path,
            )

            model.to(self.device)
            model.eval()

            self.models.append(model)

    @staticmethod
    def _create_model() -> nn.Module:
        """
        Must exactly match the training architecture.
        """

        return DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
        )

    @staticmethod
    def _load_checkpoint(
        model: nn.Module,
        checkpoint_path: Path,
    ) -> None:

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        # Standard case:
        # torch.save(model.state_dict(), path)
        if isinstance(checkpoint, dict):

            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

            else:
                state_dict = checkpoint

        else:
            raise ValueError(
                f"Unsupported checkpoint format: "
                f"{checkpoint_path}"
            )

        # Remove common prefixes if the model was saved
        # through DataParallel / wrappers.
        cleaned_state_dict = {}

        for key, value in state_dict.items():

            if key.startswith("module."):
                key = key[len("module."):]

            if key.startswith("model."):
                key = key[len("model."):]

            cleaned_state_dict[key] = value

        missing, unexpected = model.load_state_dict(
            cleaned_state_dict,
            strict=False,
        )

        if missing:
            raise RuntimeError(
                f"Missing keys when loading "
                f"{checkpoint_path}:\n"
                f"{missing}"
            )

        if unexpected:
            raise RuntimeError(
                f"Unexpected keys when loading "
                f"{checkpoint_path}:\n"
                f"{unexpected}"
            )

    @staticmethod
    def _create_transforms():

        return Compose([

            LoadImage(
                image_only=True,
            ),

            EnsureChannelFirst(),

            Orientation(
                axcodes="RAS",
            ),

            Spacing(
                pixdim=TARGET_SPACING,
                mode="bilinear",
            ),
            CropForeground(),

            NormalizeIntensity(
                nonzero=True,
            ),

            

            ResizeWithPadOrCrop(
                spatial_size=TARGET_SIZE,
            ),

            
        ])
        

    def predict_one(self,nifti_path: Path,) -> float:
        """
        Predicts probabilities and averages predictions when several checkpoints are provided
        """

        transforms = self._create_transforms()

        image = transforms(str(nifti_path))

        # image shape:
        # [1, 112, 112, 80]
        image = torch.as_tensor(
            np.asarray(image),
            dtype=torch.float32)

        # Add batch dimension:
        # [1, 1, 112, 112, 80]
        image = image.unsqueeze(0)

        image = image.to(self.device)

        predictions = []

        with torch.no_grad():

            for model in self.models:

                logits = model(image)
                probability = torch.sigmoid(logits)
                predictions.append(probability.item())

                #predictions.append(logits.squeeze().item())
                #logits = model(image)
                #probabilities.append(torch.sigmoid(logits).squeeze().item())

        
        return float(np.mean(predictions))

    def predict(
        self,
        nifti_paths: Iterable[Path],
    ) -> list[float]:

        return [
            self.predict_one(path)
            for path in nifti_paths
        ]


