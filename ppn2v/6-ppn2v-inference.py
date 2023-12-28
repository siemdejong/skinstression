import warnings

warnings.filterwarnings("ignore")
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from tifffile import imread, imwrite

from ppn2v.pn2v import gaussianMixtureNoiseModel, histNoiseModel, prediction, utils
from ppn2v.pn2v.utils import PSNR

# See if we can use a GPU
device = utils.getDevice()

# We need the training data in order to calculate 'mean' and 'std' for normalization
path = "/home/sdejong/skinstression/data/stacks/"
output_path = "/home/sdejong/skinstression/data/stacks-denoised/"

dataName = "skin"
nameNoiseModel = (
    "GMMNoiseModel_" + dataName + "_" + str(3) + "_" + str(2) + "_" + "bootstrap"
)

namePN2VModel = nameNoiseModel
net = torch.load(path + "/last_" + namePN2VModel + ".net")
if "HistNoiseModel" in namePN2VModel:
    histogram = np.load(path + nameNoiseModel + ".npy")
    noiseModel = histNoiseModel.NoiseModel(histogram, device=device)
elif "GMMNoiseModel" in namePN2VModel:
    params = np.load(path + nameNoiseModel + ".npz")
    noiseModel = gaussianMixtureNoiseModel.GaussianMixtureNoiseModel(
        params=params, device=device
    )

for img_path in glob(path + "*.tif"):

    results = []

    data = imread(img_path)

    for idx in range(data.shape[0]):
        im = data[idx]
        _, mseEst = prediction.tiledPredict(
            im, net, ps=192, overlap=48, device=device, noiseModel=noiseModel
        )

        results.append(mseEst)

    imwrite(output_path + img_path.split("/")[-1], results)

    break
