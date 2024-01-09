# image-classification
This repository is dedicated to solving image classification task through convolutional neuron networks using CIFAR10 dataset. The architecture of model is similar to one block of ResNet 

<img src="https://i.ibb.co/2hg962h/basic-block.png" width="300"/>

- Every convolution layer got 32 neurons
- block's ouput is reduced to size 32x4x4 using average pooling
- to get logits after average pooling the ouput is flatten and pushed as input to linear layer

## Results
  
![image](https://github.com/EdA20/image-classification/assets/64848449/0871e2f4-cef1-4148-a8b9-846898e47786)

Acuraccy increases on train and test what indicates that model is not overfitted. On 20 epochs test accuracy reaches 73%

### Futher improves
- Add augmentation
- Add scheduler in training process
- Replace Average Pooling with Max Pooling
- Replace ReLU with Leaky ReLU

