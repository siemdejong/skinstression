import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from tifffile import imread

from ppn2v.pn2v import training, utils
from ppn2v.unet import UNet

# See if we can use a GPU
device = utils.getDevice()

path = "/home/sdejong/skinstression/data/stacks/"
fileName = "17.tif"
dataName = "skin"  # This will be used to name the noise2void model

data = imread(path + fileName)
nameModel = dataName + "_n2v"

# The N2V network requires only a single output unit per pixel
net = UNet(1, depth=3)

# Split training and validation data.
my_train_data = data[:-5].copy()
my_val_data = data[-5:].copy()

# Start training.
trainHist, valHist = training.trainNetwork(
    net=net,
    trainData=my_train_data,
    valData=my_val_data,
    postfix=nameModel,
    directory=path,
    noiseModel=None,
    device=device,
    numOfEpochs=200,
    stepsPerEpoch=10,
    virtualBatchSize=20,
    batchSize=1,
    learningRate=1e-3,
)

# Let's look at the training and validation loss
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(valHist, label="validation loss")
plt.plot(trainHist, label="training loss")
plt.legend()
plt.savefig("n2v-training.png")
