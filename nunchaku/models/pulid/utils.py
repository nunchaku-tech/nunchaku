"""
This module provides utility functions for PuLID.

.. note::
    This module is adapted from the original PuLID repository:
    https://github.com/ToTheBeginning/PuLID
"""

import math

import cv2
import numpy as np
import torch
from torchvision.utils import make_grid


def resize_numpy_image_long(image, resize_long_edge=768):
    """
    Resize a numpy image so that its longest edge matches ``resize_long_edge``, preserving aspect ratio.

    If the image's longest edge is already less than or equal to ``resize_long_edge``, the image is returned unchanged.

    Parameters
    ----------
    image : np.ndarray
        Input image as a numpy array of shape (H, W, C).
    resize_long_edge : int, optional
        Desired size for the longest edge (default: 768).

    Returns
    -------
    np.ndarray
        The resized image as a numpy array.
    """
    h, w = image.shape[:2]
    if max(h, w) <= resize_long_edge:
        return image
    k = resize_long_edge / max(h, w)
    h = int(h * k)
    w = int(w * k)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """
    Convert numpy images to PyTorch tensors.

    This function supports both single images and lists of images. The images are converted from
    HWC (height, width, channel) format to CHW (channel, height, width) format. Optionally, BGR images
    can be converted to RGB, and the output can be cast to float32.

    Parameters
    ----------
    imgs : np.ndarray or list of np.ndarray
        Input image(s) as numpy array(s).
    bgr2rgb : bool, optional
        Whether to convert BGR images to RGB (default: True).
    float32 : bool, optional
        Whether to cast the output tensor(s) to float32 (default: True).

    Returns
    -------
    torch.Tensor or list of torch.Tensor
        Converted tensor(s). If a single image is provided, returns a tensor; if a list is provided, returns a list of tensors.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == "float64":
                img = img.astype("float32")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """
    Convert PyTorch tensor(s) to image numpy array(s).

    This function supports 4D mini-batch tensors, 3D tensors, and 2D tensors. The output is a numpy array
    in HWC (height, width, channel) or HW (height, width) format. Optionally, RGB images can be converted to BGR,
    and the output type can be specified.

    After clamping to [min, max], values are normalized to [0, 1].

    Parameters
    ----------
    tensor : torch.Tensor or list of torch.Tensor
        Input tensor(s). Accepts:

        1) 4D mini-batch tensor of shape (B x 3/1 x H x W)
        2) 3D tensor of shape (3/1 x H x W)
        3) 2D tensor of shape (H x W)

        The channel order should be RGB.

    rgb2bgr : bool, optional
        Whether to convert RGB images to BGR (default: True).

    out_type : numpy type, optional
        Output data type. If ``np.uint8``, output is in [0, 255]; otherwise, in [0, 1] (default: np.uint8).

    min_max : tuple of int, optional
        Min and max values for clamping (default: (0, 1)).

    Returns
    -------
    np.ndarray or list of np.ndarray
        Converted image(s) as numpy array(s). If a single tensor is provided, returns a numpy array; if a list is provided, returns a list of numpy arrays.

    Raises
    ------
    TypeError
        If the input is not a tensor or list of tensors, or if the tensor has unsupported dimensions.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f"Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}")
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result
