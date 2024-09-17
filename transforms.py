from __future__ import annotations

from typing import Callable
import torch
from torch import Tensor, nn
from torchaug.batch_transforms import (
    BatchVideoWrapper,
    BatchRandomColorJitter,
    BatchRandomGaussianBlur,
    BatchRandomGrayScale,
    BatchRandomSolarize
)
from torchaug.transforms import VideoNormalize
from torchvision.transforms import Compose, InterpolationMode, RandomResizedCrop, RandomHorizontalFlip
import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode


class RandomHorizontalFlipSample(RandomHorizontalFlip):
    def __init__(self, p: float) -> None:
        super().__init__(p)
    
    def forward(self, sample: dict[str: Tensor]) -> dict[str: Tensor]:
        if torch.rand(1) < self.p:
            sample["image"] =  F.hflip(sample["image"])
            sample["mask"] =  F.hflip(sample["mask"])
            sample["flipped"] = torch.Tensor([True])
        else:
            sample["flipped"] = torch.Tensor([False])
        return sample


class RandomResizedCropSample(RandomResizedCrop):
    def __init__(self, size, scale=..., ratio=..., interpolation=InterpolationMode.BILINEAR, antialias=True, interpolation_mask=InterpolationMode.NEAREST, p:float =0.):
        if size is None:
            size = 4
            none_size = True
        else:
            none_size = False

        super().__init__(size, scale, ratio, interpolation, antialias)

        if none_size:
            self.size = None
        self.p=p
        self.interpolation_mask = interpolation_mask

    def forward(self, sample: dict[str: Tensor]) -> dict[str: Tensor]:
        if torch.rand(1) < self.p:
            if self.size is None:
                size: list[int] = sample["image"].shape[-2:]
            else:
                size = self.size
            i, j, h, w = self.get_params(sample["image"], self.scale, self.ratio)
            sample["image"] =  F.resized_crop(sample["image"], i, j, h, w, size, self.interpolation, antialias=self.antialias)
            sample["mask"] =  F.resized_crop(sample["mask"], i, j, h, w, size, self.interpolation_mask, antialias=self.antialias)
            sample["cropped"] = torch.Tensor([True])
        else:
            sample["cropped"] = torch.Tensor([False])
        return sample

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        interpolate_mask_str = self.interpolation_mask.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str}"
        format_string += f", interpolation_mask={interpolate_mask_str}"
        format_string += f", antialias={self.antialias}"
        format_string += f", p={self.p})"
        return format_string


def get_crop_transform(
    size: int | list[int] | None = None,
    scale: list[float] = (0.08, 1.0),
    ratio: list[float] = (3.0 / 4.0, 4.0 / 3.0),
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    interpolation_mask: InterpolationMode = InterpolationMode.NEAREST,
    p_crop: float = 0.,
    p_flip: float = 0.5,
) -> Callable:
    return Compose(
        [
            RandomResizedCropSample(size, scale, ratio, interpolation, True, interpolation_mask, p_crop),
            RandomHorizontalFlipSample(p_flip)
        ]
    )


def get_ssl_train_online_transform(
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.4,
    hue: float = 0.1,
    p_contrastive: float = 0.8,
    p_grayscale: float = 0.2,
    kernel_size_blur: int = 23,
    sigma_blur: list[int] = [0.1, 2.],
    p_blur: float = 0.5,
    threshold_solarize: float = 0.5,
    p_solarize: float = 0.2,
    mean: list[int] = [0.485, 0.456, 0.406],
    std: list[int] = [0.229, 0.224, 0.225]
) -> nn.Module:
    return BatchVideoWrapper(
        [
            BatchRandomColorJitter(brightness, contrast, saturation, hue, p_contrastive),
            BatchRandomGrayScale(p_grayscale),
            BatchRandomGaussianBlur(kernel_size_blur, sigma_blur, p_blur),
            BatchRandomSolarize(threshold_solarize, p_solarize),
            VideoNormalize(mean, std, video_format="TCHW")
        ],
        inplace=False,
        same_on_frames=True,
        video_format="TCHW",
    )


def get_normalize_transform(
    mean: list[int] = [0.485, 0.456, 0.406],
    std: list[int] = [0.229, 0.224, 0.225]
) -> nn.Module:
    return VideoNormalize(mean, std, inplace=False, video_format="TCHW")
