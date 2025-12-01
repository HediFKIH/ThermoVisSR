import torch
import torch.nn as nn
from torchvision import models
import timm
from typing import Optional
import warnings

from utils import MeanShift, MeanShift4


class Vgg19(nn.Module):
    """
    VGG19 feature extractor for visible (RGB) images.
    
    Extracts features up to relu5_1 layer for perceptual loss computation.
    Pretrained on ImageNet and uses normalized inputs.
    
    Args:
        requires_grad (bool): If True, allows gradient computation through VGG layers.
                             Default: False (frozen features).
        rgb_range (float): Range of input RGB values. Default: 1 (normalized to [0,1]).
        device (str): Device to load the model on ('cuda' or 'cpu').
    
    Attributes:
        slice1 (nn.Sequential): VGG19 layers up to relu5_1.
        sub_mean (MeanShift): Layer for mean subtraction and std normalization.
    
    Example:
        >>> extractor = Vgg19FeatureExtractor(requires_grad=False, rgb_range=1)
        >>> features = extractor(rgb_image)  # Extract features
    """
    
    def __init__(
        self, 
        requires_grad: bool = False, 
        rgb_range: float = 1.0,
        device: Optional[str] = None
    ):
        super(Vgg19, self).__init__()
        
        # Load pretrained VGG19
        try:
            vgg_pretrained = models.vgg19(pretrained=True)
        except:
            # Support for newer PyTorch versions
            vgg_pretrained = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        vgg_features = vgg_pretrained.features
        
        # Extract features up to relu5_1 (layer 30)
        self.slice1 = nn.Sequential()
        for x in range(30):
            self.slice1.add_module(str(x), vgg_features[x])
        
        # Freeze VGG parameters if not training
        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = False
        
        # VGG normalization parameters (ImageNet statistics)
        vgg_mean = (0.5710, 0.6199, 0.6584)
        vgg_std = (
            0.0851 * rgb_range, 
            0.0720 * rgb_range, 
            0.0924 * rgb_range
        )
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)
        
        # Move to device if specified
        if device:
            self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through VGG19 feature extractor.
        
        Args:
            x (torch.Tensor): Input RGB image tensor of shape (B, 3, H, W).
        
        Returns:
            torch.Tensor: Extracted features of shape (B, 512, H/16, W/16).
        """
        # Normalize input
        h = self.sub_mean(x)
        
        # Extract relu5_1 features
        h_relu5_1 = self.slice1(h)
        
        return h_relu5_1
    
    def get_num_params(self) -> int:
        """Returns the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self) -> int:
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Vgg19thr(nn.Module):
    """
    VGG19 feature extractor for thermal (IR) images with 4 channels.
    
    Modified VGG19 that accepts 4-channel input (RGB + Thermal).
    Extracts features up to relu5_1 layer for perceptual loss computation.
    
    Args:
        requires_grad (bool): If True, allows gradient computation through VGG layers.
                             Default: False (frozen features).
        rgb_range (float): Range of input values. Default: 1 (normalized to [0,1]).
        device (str): Device to load the model on ('cuda' or 'cpu').
    
    Attributes:
        slice1 (nn.Sequential): Modified VGG19 layers up to relu5_1.
        sub_mean (MeanShift4): Layer for 4-channel mean subtraction and normalization.
    
    Example:
        >>> extractor = Vgg19ThermalFeatureExtractor(requires_grad=False)
        >>> features = extractor(thermal_image)  # Extract features from 4-channel input
    """
    
    def __init__(
        self, 
        requires_grad: bool = False, 
        rgb_range: float = 1.0,
        device: Optional[str] = None
    ):
        super(Vgg19thr, self).__init__()
        
        # Load VGG19 with 4 input channels using timm
        try:
            vgg_pretrained = timm.create_model(
                'vgg19', 
                pretrained=True, 
                in_chans=4
            )
        except Exception as e:
            warnings.warn(
                f"Failed to load pretrained VGG19 with 4 channels: {e}\n"
                f"Loading standard VGG19 and adapting first layer..."
            )
            vgg_pretrained = models.vgg19(pretrained=True)
            # Modify first conv layer to accept 4 channels
            old_conv = vgg_pretrained.features[0]
            new_conv = nn.Conv2d(
                4, 64, 
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding
            )
            # Initialize new channel with mean of RGB weights
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = old_conv.weight
                new_conv.weight[:, 3:, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
                new_conv.bias = old_conv.bias
            vgg_pretrained.features[0] = new_conv
        
        vgg_features = vgg_pretrained.features
        
        # Extract features up to relu5_1 (layer 30)
        self.slice1 = nn.Sequential()
        for x in range(30):
            self.slice1.add_module(str(x), vgg_features[x])
        
        # Freeze VGG parameters if not training
        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = False
        
        # VGG normalization parameters for 4 channels (RGB + Thermal)
        # Thermal channel statistics are empirically determined
        vgg_mean = (0.5710, 0.6199, 0.6584, 0.1756)
        vgg_std = (
            0.0851 * rgb_range, 
            0.0720 * rgb_range, 
            0.0924 * rgb_range, 
            0.0991 * rgb_range
        )
        self.sub_mean = MeanShift4(rgb_range, vgg_mean, vgg_std)
        
        # Move to device if specified
        if device:
            self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through thermal VGG19 feature extractor.
        
        Args:
            x (torch.Tensor): Input 4-channel tensor of shape (B, 4, H, W).
                             Channels: [R, G, B, Thermal]
        
        Returns:
            torch.Tensor: Extracted features of shape (B, 512, H/16, W/16).
        """
        # Normalize input
        h = self.sub_mean(x)
        
        # Extract relu5_1 features
        h_relu5_1 = self.slice1(h)
        
        return h_relu5_1
    
    def get_num_params(self) -> int:
        """Returns the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self) -> int:
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

