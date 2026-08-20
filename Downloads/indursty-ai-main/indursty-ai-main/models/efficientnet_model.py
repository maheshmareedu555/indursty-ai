import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2

class EfficientNetB3Inspector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetB3Inspector, self).__init__()
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b3(weights=weights)
        
        # Replace classification head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
        # Target layer for Grad-CAM (last convolutional layer of features)
        self.target_layer = self.backbone.features[-1]
        self.gradients = None
        self.activations = None
        
        # Register hooks for Grad-CAM
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        return self.backbone(x)

    def generate_gradcam(self, input_tensor, target_class=None):
        """
        Generates Grad-CAM visual heatmap overlay for defect localization.
        """
        self.eval()
        output = self.forward(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        self.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
            
        return cam, target_class, torch.softmax(output, dim=1)[0].detach().cpu().numpy()

def get_transforms():
    train_transform = T.Compose([
        T.Resize((300, 300)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = T.Compose([
        T.Resize((300, 300)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform
