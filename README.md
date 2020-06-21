## WideResNet22 - PyTorch

### Model with Residual Blocks and Batch Normalization
![Residual Block](https://cv-tricks.com/wp-content/uploads/2019/07/Simple_Residual_Block.png)

We'll use a significatly larger model this time, called the WideResNet22, which has 22 convolutional layers. However, one of the key changes to our model is the addition of the **resudial block**, which adds the original input back to the output feature map obtained by passing the input through one or more convolutional layers.

Here we're trying to build a deep residual neural network to classify images from the CIFAR10 dataset with around 90+ accuracy. In this project, we'll use the following techniques to achieve SOTA accuracy in less than 10 minutes:

-   Data normalization
-   Data augmentation
-   Residual connections
-   Batch normalization
-   Learning rate annealing
-   Weight Decay
-   Gradient clipping
### Technologies Used
- PyTorch
- FastAI

### Training

    python train.py
