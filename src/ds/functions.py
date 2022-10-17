import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x, A, h, slope, C) -> float:
    """https://stackoverflow.com/a/55104465"""
    return 1 / (1 + np.exp((x - h) / slope)) * A + C
