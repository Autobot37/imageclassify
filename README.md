# ML for Sustainibility

# The following repo covers


# Data Utilities

- Data augmentation and transforms to prevent overfitting with e.g., random flip.
- Image size we resizing is (224,224,3).
- Normalizing to a mean and standard deviation. I specifically choose this mean because it's the mean and std from the ImageNet dataset and a good representative for image datasets ([source](https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2)).

## Train Function

- For every epoch, it first trains on `train_loader` and does validation on `val_loader`.
- Using `tqdm` for a progress bar.
- Default hyperparameters:
  - Epochs per fold: 3
  - Learning rate: 1e-4
  - Loss function: CrossEntropyLoss
  - Optimizer: AdamW with lr = 1e-4 and defaults betas (0.9,0.99) with 0.01 default weight decay.

[Superconvergence post](https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html)

- Optimizations like:
  - Deleting model at the end and saving best weights.
  - Deleting images and labels sent to GPU at the end.
  - Half precision for validation, since we don't have to update weights here, using half precision makes it fast for fast evaluating.

## Main One vs Rest Binary Classification Function

- Here we will select some class and then perform binary classification on it, meaning we assign 1 label to the selected class and 0 label to other classes and then take loss by CrossEntropy and perform weight update.
- For every data class, we will make a train and Validation Dataloader for 3 folds and collect data like loss and accuracy to plot.

## Dataloader

- Taking images from a particular data index class and other data classes, and for binary classification, we set data index class labels as 1 and other labels as 0.
- Here `take` is a parameter which defines how much to take from other classes. Currently, we take 10% of other classes and it works.
- Setting number of workers 2 and pin memory for faster data caching.
- `SubsetRandomSampler` for selecting indices.

## Batch Size

- Batch size is 32, since GPU we are using P100 has enough memory and bottleneck is not GPU but CPU for data caching.

## CustomModel

- We used model with AttentionBlock to emphasize the importance of image patches, to focus on important part of images and discard irrelevant parts. Although it is helpful in medical datasets but in animal datasets maybe there is a big landscape with animal in small pixels for e.g., with birds, there attention will be useful.

### Components

**Convolution Blocks**

- There are 5 Conv Blocks each constituting:
  - `conv2d`, `batchnorm`, `relu`, `conv2d`, `batchnorm`, `relu` and `maxpool` at the end.
- These convblocks are commonly used.
- `BatchNorm` is used to stabilize training from covariate shift ([source](https://arxiv.org/abs/1502.03167)).
- `ReLU` is used as an activation function as used by ImageNet paper.

**Hyperparameters**

- Kernel size: 3 for Conv filters and 2 for MaxPooling.

**Attention Block**

- The intermediate features is the output of pool-3 or pool-4 and the global feature vector (output of pool-5) is fed as input to the attention layer.
- Feature upsampling is done via bilinear interpolation to make intermediate and global feature vector same shape.
- After that an element-wise sum is done followed by a convolution operation that just reduces the 256 channels to 1.
- This is then fed into a Softmax layer, which gives us a normalized Attention map (A). Each scalar element in A represents the degree of attention to the corresponding spatial feature vector in F.
- The new feature vector 𝐹̂ is then computed by pixel-wise multiplication. That is, each feature vector f is multiplied by the attention element a
- So, the attention map A and the new feature vector 𝐹̂ are the outputs of the Attention Layer.

[Reference](https://github.com/SaoYan/IPMI2019-AttnMel/blob/99e4a9b71717fb51f24d7994948b6a0e76bb8d58/networks.py)

## Initializing

- We initialize with Kaiming normal to prevent gradient vanishing or exploding latter on.
  - `Conv2d` Layers: initializes the weights using the Kaiming normal.
  - `BatchNorm2d` Layers: it sets the weights to 1 and biases to 0.
  - `Linear` Layers: initializes the weights from a normal distribution with mean 0 and standard deviation 0.01. It sets the biases to 0.




## At the end we achieved high accuracy for Resnet EfficientNet and Custom Convolution Attention Model

# How to run this 
- since it has dataset requirements it can be run in kaggle
- download [https://github.com/Autobot37/imageclassify/blob/main/nbs/ml-for-sustainability-srip-2024.ipynb] and import notebook there and run it.

# Author ; Shiva Singh Bagri
