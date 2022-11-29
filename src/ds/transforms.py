from scipy import stats
import torch
from torchvision.utils import _log_api_usage_once
import numpy as np
from PIL import Image


class YeoJohnsonTransform(torch.nn.Module):
    def __init__(self, lmbda: float):
        super().__init__()
        _log_api_usage_once(self)
        self.lmbda = lmbda

    def forward(self, img):
        img = np.asarray(img)
        img_dtype = img.dtype
        transformed_image = stats.yeojohnson(img, self.lmbda)
        img_out = transformed_image.astype(img_dtype)

        return img_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lmbda={self.lmbda})"
