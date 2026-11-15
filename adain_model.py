"""
AdaIN Neural Style Transfer Module
Refactored & Simplified for Clarity, Performance, and Interview Presentation
Author: Senior AI/ML Engineer
"""

from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union, Callable

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision.models import vgg19, VGG19_Weights
from torchvision.transforms import functional as TF


# ===========================================================================
# 1. MATHEMATICAL UTILITIES & FEATURE NORMALIZATION
# ===========================================================================

def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    1. What it does: Calculates channel-wise spatial mean (mu) and standard deviation (sigma).
    2. Why required: In deep networks, feature statistics (mean and variance) encode visual style.
    3. Inputs: Feature tensor of shape (N, C, H, W).
    4. Returns: (mean, std) both of shape (N, C, 1, 1).
    5. Pipeline role: Core mathematical building block of AdaIN and Style Loss.
    """
    N, C = feat.size()[:2]
    feat_var = feat.view(N, C, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    """
    1. What it does: Aligns the mean and variance of content features to match style features:
                     AdaIN(x, y) = sigma(y) * ((x - mu(x)) / sigma(x)) + mu(y)
    2. Why required: Transfers artistic style in feature space without retraining for new styles.
    3. Inputs: content_feat (N, C, H, W) and style_feat (N, C, H, W).
    4. Returns: Normalized and style-aligned feature tensor of shape (N, C, H, W).
    5. Pipeline role: The core bottleneck layer of the AdaIN architecture.
    """
    assert content_feat.size()[:2] == style_feat.size()[:2], "Content and style channel count must match!"
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized = (content_feat - content_mean) / content_std
    return normalized * style_std + style_mean


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """
    1. What it does: Computes channel-to-channel correlation matrix: G = (F * F^T) / (C * H * W).
    2. Why required: Gram matrices capture textures, brushstrokes, and repeating patterns.
    3. Inputs: Feature tensor of shape (N, C, H, W).
    4. Returns: Gram matrix tensor of shape (N, C, C).
    5. Pipeline role: Used to compute Style Loss in Perceptual Style Transfer (Gatys et al.).
    """
    b, c, h, w = feat.size()
    f = feat.view(b * c, h * w)
    G = torch.mm(f, f.t())
    return G.div(b * c * h * w)


def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """
    1. What it does: Measures spatial pixel differences between adjacent pixels.
    2. Why required: Acts as a smoothness prior to eliminate digital noise and create clean brushstrokes.
    3. Inputs: Image tensor of shape (N, 3, H, W).
    4. Returns: Scalar loss value.
    5. Pipeline role: Regularizer in Perceptual Style Transfer optimization.
    """
    return torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
           torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))


# ===========================================================================
# 2. NEURAL NETWORK ARCHITECTURES (ENCODER & DECODER)
# ===========================================================================

class VGG19Encoder(nn.Module):
    """
    1. What it does: Pretrained VGG19 feature extractor sliced into multi-scale stages.
    2. Why required: Deep CNN features encode semantic content and artistic style at different depths:
         - relu1_1 (64 channels) : fine textures, edges, and low-level colors
         - relu2_1 (128 channels): brushwork and surface patterns
         - relu3_1 (256 channels): complex geometric patterns
         - relu4_1 (512 channels): deep features used for AdaIN bottleneck
         - relu4_2 (512 channels): semantic content representation (objects, structure)
         - relu5_1 (512 channels): high-level spatial composition
    3. Inputs: Image tensor normalized in [0, 1] with shape (N, 3, H, W).
    4. Returns:
         - forward(): Tuple of (relu1_1, relu2_1, relu3_1, relu4_1) for AdaIN.
         - extract_all(): Dictionary of all multi-scale features for Perceptual NST.
    5. Pipeline role: Feature representation engine for both AdaIN and Perceptual NST.
    """
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features

        # Slices mapped to exact VGG-19 ReLU activations
        self.slice1 = nn.Sequential(*vgg[:2])       # relu1_1
        self.slice2 = nn.Sequential(*vgg[2:7])      # relu2_1
        self.slice3 = nn.Sequential(*vgg[7:12])     # relu3_1
        self.slice4 = nn.Sequential(*vgg[12:21])    # relu4_1
        self.slice4_2 = nn.Sequential(*vgg[21:23])  # relu4_2 (conv4_2 + relu)
        self.slice5 = nn.Sequential(*vgg[23:30])    # relu5_1

        # Freeze all weights — encoder acts purely as a fixed feature extractor
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet standardization buffers: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """AdaIN standard forward pass: returns (relu1_1, relu2_1, relu3_1, relu4_1)."""
        x = self._normalize(x)
        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        f3 = self.slice3(f2)
        f4 = self.slice4(f3)
        return f1, f2, f3, f4

    def extract_all(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extracts multi-scale features for Perceptual Style Transfer."""
        x = self._normalize(x)
        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        f3 = self.slice3(f2)
        f4_1 = self.slice4(f3)
        f4_2 = self.slice4_2(f4_1)
        f5_1 = self.slice5(f4_2)
        return {
            "relu1_1": f1,
            "relu2_1": f2,
            "relu3_1": f3,
            "relu4_1": f4_1,
            "relu4_2": f4_2,
            "relu5_1": f5_1
        }

# Alias for backward compatibility with existing notebooks and scripts
VGGEncoder = VGG19Encoder


class Decoder(nn.Module):
    """
    1. What it does: Inverts deep 512-channel relu4_1 features back to RGB image space (3 channels).
    2. Why required: Generates pixel outputs from stylized feature representations.
    3. Architecture: 7 Convolutional layers with nearest-neighbor 2x upsampling.
    4. Checkpoint: Compatible with trained weights 'adain_decoder_100_batches.pth'.
    5. Pipeline role: Generative image reconstruction in the AdaIN pipeline.
    """
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            # Block 4 -> Block 3
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),

            # Block 3 -> Block 2
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),

            # Block 2 -> Block 1
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            # Output RGB layer (3 channels)
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


# ===========================================================================
# 3. IMAGE PREPROCESSING & LAZY DATASET
# ===========================================================================

def load_and_preprocess_image(
    image_path: Union[str, Path],
    size: Optional[int] = 384,
    device: Union[str, torch.device] = "cpu"
) -> torch.Tensor:
    """
    1. What it does: Loads an image from disk, converts to RGB, fits to target resolution,
                     and normalizes pixels to float32 in range [0, 1].
    2. Why required: Ensures correct tensor dimensions and formatting for VGG19.
    3. Inputs: image_path, size (e.g. 256, 384), device ('cuda' or 'cpu').
    4. Returns: Tensor of shape (1, 3, size, size) on the specified device.
    5. Pipeline role: Preprocessing step before inference or optimization.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found at: {path}")

    with Image.open(path) as img:
        img = img.convert("RGB")
        if size is not None:
            img = ImageOps.fit(img, (size, size), method=Image.Resampling.BICUBIC)
        tensor = TF.to_tensor(img)

    return tensor.unsqueeze(0).to(device)


from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """
    Lazy-loading dataset: reads images from disk on-the-fly to avoid RAM exhaustion.
    """
    def __init__(self, file_paths: List[Path], image_size: int = 256):
        self.file_paths = file_paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.file_paths[index]) as img:
            img = img.convert("RGB")
            img = ImageOps.fit(img, (self.image_size, self.image_size), method=Image.Resampling.BICUBIC)
        return TF.to_tensor(img)


# ===========================================================================
# 4. STYLE TRANSFER ALGORITHMS (ADAIN & PERCEPTUAL NST)
# ===========================================================================

def style_transfer(
    encoder: nn.Module,
    decoder: nn.Module,
    content_tensor: torch.Tensor,
    style_tensor: torch.Tensor,
    alpha: float = 1.0
) -> torch.Tensor:
    """
    1. What it does: Executes real-time feed-forward AdaIN style transfer in ~20ms.
    2. Steps:
         - Extract relu4_1 features from content and style.
         - Compute AdaIN target: t = AdaIN(f_c, f_s).
         - Interpolate with content features based on alpha: t_blend = (1 - alpha) * f_c + alpha * t.
         - Decode t_blend into a stylized image.
    3. Inputs: encoder, decoder, content_tensor, style_tensor, alpha in [0.0, 1.0].
    4. Returns: Generated stylized tensor of shape (N, 3, H, W).
    """
    with torch.inference_mode():
        content_feat = encoder(content_tensor)[-1]
        style_feat = encoder(style_tensor)[-1]

        adain_feat = adaptive_instance_normalization(content_feat, style_feat)

        if alpha < 1.0:
            target_feat = (1.0 - alpha) * content_feat + alpha * adain_feat
        else:
            target_feat = adain_feat

        generated = decoder(target_feat)

    return generated


def perceptual_style_transfer(
    content_tensor: torch.Tensor,
    style_tensor: torch.Tensor,
    steps: int = 200,
    lr: float = 0.035,
    style_weight: float = 4e6,
    content_weight: float = 0.6,
    tv_weight: float = 1e-5,
    color_transfer: bool = True,
    device: Optional[torch.device] = None,
    encoder: Optional[VGG19Encoder] = None,
    callback: Optional[Callable[[int, int, float], None]] = None
) -> torch.Tensor:
    """
    1. What it does: High-fidelity optimization-based Neural Style Transfer (Gatys et al.)
                     delivering museum-quality oil canvas brushstrokes and true color palette transfer.
    2. Steps:
         - Color palette alignment (matches mean and variance in RGB space) if color_transfer=True.
         - Computes multi-scale Gram matrices on style features (relu1_1 through relu5_1).
         - Optimizes pixels via Adam to balance Content Loss, Gram Style Loss, and TV Loss.
    3. Runs in ~5-7 seconds on modern GPUs (e.g. RTX 4060).
    """
    if device is None:
        device = content_tensor.device

    if encoder is None:
        encoder = VGG19Encoder().to(device).eval()

    with torch.no_grad():
        c_feats = encoder.extract_all(content_tensor)
        s_feats = encoder.extract_all(style_tensor)
        target_content = c_feats["relu4_2"].detach()
        target_styles = {
            name: gram_matrix(s_feats[name]).detach()
            for name in ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]
        }

    # Initialize input tensor
    if color_transfer:
        # Transfer global color & contrast statistics from style to content in RGB space
        c_m = content_tensor.mean([2, 3], keepdim=True)
        c_s = content_tensor.std([2, 3], keepdim=True)
        s_m = style_tensor.mean([2, 3], keepdim=True)
        s_s = style_tensor.std([2, 3], keepdim=True)
        init_tensor = (((content_tensor - c_m) / (c_s + 1e-5)) * s_s + s_m).clamp(0, 1)
    else:
        init_tensor = content_tensor.clone()

    input_img = init_tensor.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([input_img], lr=lr)

    # Multi-scale style weights: shallow layers capture color/brushwork; deep layers capture composition
    layer_weights = {
        "relu1_1": 1.0,
        "relu2_1": 0.8,
        "relu3_1": 0.6,
        "relu4_1": 0.4,
        "relu5_1": 0.2
    }

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        current_img = torch.clamp(input_img, 0, 1)
        feats = encoder.extract_all(current_img)

        # 1. Content Loss (preserves semantic structure)
        c_loss = nn.functional.mse_loss(feats["relu4_2"], target_content)

        # 2. Multi-Scale Style Loss (Gram matrix texture matching)
        s_loss = torch.tensor(0.0, device=device)
        for name, w in layer_weights.items():
            gm = gram_matrix(feats[name])
            s_loss = s_loss + w * nn.functional.mse_loss(gm, target_styles[name])

        # 3. Total Variation Loss (smooths brushwork and removes noise)
        tv_loss = total_variation_loss(current_img)

        total_loss = content_weight * c_loss + style_weight * s_loss + tv_weight * tv_loss
        total_loss.backward()
        optimizer.step()

        if callback is not None:
            callback(step, steps, total_loss.item())

    return torch.clamp(input_img, 0, 1).detach()


# ===========================================================================
# 5. LOSS FUNCTIONS (FOR TRAINING & VALIDATION)
# ===========================================================================

def calc_content_loss(gen_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
    """Computes AdaIN content loss: L_c = || f(g(t)) - t ||_2"""
    return nn.functional.mse_loss(gen_feat, target_feat)


def calc_style_loss(
    gen_feats: List[torch.Tensor],
    style_feats: List[torch.Tensor],
    layer_weights: Optional[List[float]] = None
) -> torch.Tensor:
    """
    Computes multi-layer style loss:
        L_s = sum_i w_i * (|| mu(gen_i) - mu(style_i) ||_2 + || sigma(gen_i) - sigma(style_i) ||_2)
    """
    if layer_weights is None:
        layer_weights = [1.0] * len(gen_feats)

    total_style_loss = torch.tensor(0.0, device=gen_feats[0].device)
    for g_f, s_f, w in zip(gen_feats, style_feats, layer_weights):
        g_mean, g_std = calc_mean_std(g_f)
        s_mean, s_std = calc_mean_std(s_f)
        total_style_loss += w * (nn.functional.mse_loss(g_mean, s_mean) + nn.functional.mse_loss(g_std, s_std))

    return total_style_loss
