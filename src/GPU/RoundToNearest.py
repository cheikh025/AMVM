"""
Functions for finding nearest values
"""
import numpy as np
import torch

def FindNearest(weights: torch.tensor, nQuantized, device: torch.device):
    # Calculate the range of quantization levels
    num_levels = 2 ** nQuantized
    wMin = torch.min(weights)
    wMax = torch.max(weights)
    step = (wMax - wMin) / (2 ** nQuantized - 1)

    quantization_levels = wMin + torch.arange(num_levels, device=device) * step
    # print(quantization_levels)
    # Compute the index of the nearest quantization level for each weight
    # Using broadcasting properly to avoid incorrect dimension transformation
    indices = torch.abs(weights - quantization_levels[:, None]).argmin(dim=0)
    # Map indices to their corresponding quantization levels
    wq = quantization_levels[indices]
    return wq, indices.int()


def FindNearestNumpy(weights: np.ndarray, nQuantized):
    # Calculate the range of quantization levels
    num_levels = 2 ** nQuantized
    wMin = np.min(weights)
    wMax = np.max(weights)
    step = (wMax - wMin) / (2 ** nQuantized - 1)

    quantization_levels = wMin + np.arange(num_levels) * step
    # print(quantization_levels)
    # Compute the index of the nearest quantization level for each weight
    # Using broadcasting properly to avoid incorrect dimension transformation
    indices = np.abs(weights - quantization_levels[:, None]).argmin(axis=0)
    # Map indices to their corresponding quantization levels
    wq = quantization_levels[indices]
    return wq, indices


# TODO: make this work for full matrix, for testing and debugging purposes
def FindNearestMatrix(weights: torch.tensor, nQuantized, device: torch.device):
    """
    weights has shape (M, N)
    """
    # Calculate the range of quantization levels
    num_levels = 2 ** nQuantized
    wMin = torch.min(weights, dim=0)
    wMax = torch.max(weights, dim=0)
    step = (wMax - wMin) / (2 ** nQuantized - 1)

    quantization_levels = wMin + torch.arange(num_levels, device=device) * step
    # print(quantization_levels)
    # Compute the index of the nearest quantization level for each weight
    # Using broadcasting properly to avoid incorrect dimension transformation
    indices = torch.abs(weights - quantization_levels[:, None]).argmin(axis=0)
    # Map indices to their corresponding quantization levels
    wq = quantization_levels[indices]
    return wq, indices.int()