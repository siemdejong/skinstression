import warnings

warnings.filterwarnings("ignore")

import torch

dtype = torch.float
device = torch.device("cuda:0")

import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread

from ppn2v.pn2v import gaussianMixtureNoiseModel, histNoiseModel, prediction
from ppn2v.pn2v.utils import plotProbabilityDistribution

path = "/home/sdejong/skinstression/data/stacks/"
dataName = "skin"  # Name of the noise model
n_gaussian = 3  # Number of gaussians to use for Gaussian Mixture Model
n_coeff = 2  # No. of polynomial coefficients for parameterizing the mean, standard deviation and weight of Gaussian components

observation = imread(path + "17.tif")  # Load the appropriate data
nameHistNoiseModel = "HistNoiseModel_" + dataName + "_" + "bootstrap"
nameGMMNoiseModel = (
    "GMMNoiseModel_"
    + dataName
    + "_"
    + str(n_gaussian)
    + "_"
    + str(n_coeff)
    + "_"
    + "bootstrap"
)
nameN2VModel = dataName + "_n2v"
net = torch.load(path + "last_" + nameN2VModel + ".net")

results = []
meanRes = []
resultImgs = []
inputImgs = []
dataTest = observation

for index in range(dataTest.shape[0]):

    im = dataTest[index]
    # We are using tiling to fit the image into memory
    # If you get an error try a smaller patch size (ps)
    means = prediction.tiledPredict(
        im, net, ps=256, overlap=48, device=device, noiseModel=None
    )
    resultImgs.append(means)
    inputImgs.append(im)
    print("image:", index)

# In bootstrap mode, we estimate pseudo GT by using N2V denoised images.
signal = np.array(resultImgs)
# Let's look the raw data and our pseudo ground truth signal
print(signal.shape)
plt.figure(figsize=(12, 12))
plt.subplot(1, 2, 2)
plt.title(label="pseudo GT (generated by N2V denoising)")
plt.imshow(signal[0], cmap="gray")
plt.subplot(1, 2, 1)
plt.title(label="single raw image")
plt.imshow(observation[0], cmap="gray")
plt.show()

# We set the range of values we want to cover with our model.
# The pixel intensities in the images you want to denoise have to lie within this range.
# The dataset is clipped to values between 0 and 255.
minVal, maxVal = 234, 7402
bins = 256

# We are creating the histogram.
# This can take a minute.
histogram = histNoiseModel.createHistogram(bins, minVal, maxVal, observation, signal)

# Saving histogram to disc.
np.save(path + nameHistNoiseModel + ".npy", histogram)
histogramFD = histogram[0]

# Let's look at the histogram-based noise model
plt.xlabel("Observation Bin")
plt.ylabel("Signal Bin")
plt.imshow(histogramFD**0.25, cmap="gray")
plt.show()

min_signal = np.percentile(signal, 0.5)
max_signal = np.percentile(signal, 99.5)
print("Minimum Signal Intensity is", min_signal)
print("Maximum Signal Intensity is", max_signal)

gaussianMixtureNoiseModel = gaussianMixtureNoiseModel.GaussianMixtureNoiseModel(
    min_signal=min_signal,
    max_signal=max_signal,
    path=path,
    weight=None,
    n_gaussian=n_gaussian,
    n_coeff=n_coeff,
    device=device,
    min_sigma=50,
)
gaussianMixtureNoiseModel.train(
    signal,
    observation,
    batchSize=250000,
    n_epochs=2000,
    learning_rate=0.1,
    name=nameGMMNoiseModel,
    lowerClip=0.5,
    upperClip=99.5,
)

plotProbabilityDistribution(
    signalBinIndex=170,
    histogram=histogramFD,
    gaussianMixtureNoiseModel=gaussianMixtureNoiseModel,
    min_signal=minVal,
    max_signal=maxVal,
    n_bin=bins,
    device=device,
)
