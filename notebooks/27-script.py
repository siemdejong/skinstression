# %%
# We import all our dependencies.
from n2v.models import N2VConfig, N2V
import numpy as np
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
from matplotlib import pyplot as plt

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# %%
# We create our DataGenerator-object.
# It will help us load data and extract patches for training and validation.
datagen = N2V_DataGenerator()

# %%
# We will load all the '.png' files from the 'data' directory. In our case it is only one.
# The function will return a list of images (numpy arrays).
# In the 'dims' parameter we specify the order of dimensions in the image files we are reading:
# 'C' stands for channels (color)
imgs = datagen.load_imgs_from_directory(
    directory="/scistor/guest/sjg203/projects/shg-strain-stress/data/grayscale/z-stacks/1",
    filter="*.bmp",
    dims="YX",
)

# Let's look at the shape of the image
print("shape of loaded images: ", imgs[0].shape)
# The function automatically added an extra dimension to the image.
# It is used to hold a potential stack of images, such as a movie.

# %%
# Let's look at the image.
# We have to remove the added extra dimension to display it as 2D image.
plt.figure(figsize=(32, 16))
plt.imshow(imgs[0][0, :, :, 0])
plt.show()

# %%
# Next we extract patches for training and validation.
# The parameter 'shape' defines the size of these patches.
patch_shape = (64, 64)
patches = datagen.generate_patches_from_list(imgs, shape=patch_shape)

# %%
# Patches are created so they do not overlap.
# (Note: this is not the case if you specify a number of patches. See the docstring for details!)
# Non-overlapping patches enable us to split them into a training and validation set.
split = int(np.floor(patches.shape[0] * 0.80))
X = patches[:split]
X_val = patches[split:]

# %%
# Let's look at two patches.
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.imshow(X[0, ...])
plt.title("Training Patch")
plt.subplot(1, 2, 2)
plt.imshow(X_val[0, ...])
plt.title("Validation Patch")

# %%
# train_steps_per_epoch is set to (number of training patches)/(batch size), like this each training patch
# is shown once per epoch.
config = N2VConfig(
    X,
    unet_kern_size=3,
    unet_n_first=64,
    unet_n_depth=3,
    train_steps_per_epoch=int(X.shape[0] / 128),
    train_epochs=200,
    train_loss="mse",
    batch_norm=True,
    train_batch_size=128,
    n2v_perc_pix=0.198,
    n2v_patch_shape=(64, 64),
    n2v_manipulator="median",
    n2v_neighborhood_radius=5,
    single_net_per_channel=False,
    blurpool=True,
    skip_skipone=True,
    unet_residual=False,
)

# Let's look at the parameters stored in the config-object.
vars(config)

# %%
# a name used to identify the model --> change this to something sensible!
model_name = "n2v_sample_1"
# the base directory in which our model will live
basedir = "outputs/n2v/models"
# We are now creating our network model.
model = N2V(config, model_name, basedir=basedir)

# %%
history = model.train(X, X_val)

# %%
print(sorted(list(history.history.keys())))
plt.figure(figsize=(16, 5))
plot_history(history, ["loss", "val_loss"])

# %%
model.export_TF(
    name="Noise2Void - Skinstression - sample 1",
    description="This is the N2V for the skinstression project for sample 1",
    authors=["Siem"],
    test_img=X_val[0],
    axes="YX",
    patch_shape=patch_shape,
)
