
# Handwritten Digit Recognition using Convolutional Neural Networks with Pytorch

The MNIST dataset comprises 70,000 grayscale images of handwritten digits, each measuring 28 x 28 pixels and annotated by humans. It is a subset of a larger dataset made accessible by NIST, the National Institute of Standards and Technology. In this example, we will develop a handwritten digit recognizer utilizing a convolutional neural network (CNN) trained on the MNIST dataset, employing the PyTorch framework.

<p align="center">    
    <img width="500" aling="right" src="https://github.com/Paurnima-Chavan/MNIST-pytorch/blob/main/imgs/handwriiten.png?raw=true" />
</p>


## Documentation

This documentation provides an overview of the PyTorch model used for classifying handwritten digits from the MNIST dataset. The model architecture consists of convolutional neural networks (CNNs) and fully connected layers, which are designed to achieve accurate digit recognition.

## Code organization

Code organization in this project is structured into three files.

src

 &nbsp;&nbsp;&nbsp;&nbsp; `models.py` 

 &nbsp;&nbsp;&nbsp;&nbsp; `utils.py`

 &nbsp;&nbsp;&nbsp;&nbsp; `dataset.py`

`S6.ipynb`

The file **"models.py"** houses the class that defines the model structure. Within **"utils.py"** you'll find code responsible for training, testing, and generating performance graphs. **"dataset.py"** contains the necessary code for loading the MNIST dataset. Finally, **"S6.ipynb"** acts as the notebook where the actual execution and experimentation take place. 

This separation allows for modular and organized development of the project components.

## Model Architecture

The model, implemented in the Net class, follows a sequential structure that defines the layers and operations performed on the input data.

```bash
    class Net(nn.Module):
    """
     This defines the architecture or structure of the neural network.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.onecross1 = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2)

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.1)

        )
        self.onecross1_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 10, kernel_size=1)
        )

        self.gap1 = nn.Sequential(
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.onecross1(x)
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.onecross1_2(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gap1(x)
        x = x.squeeze()
        return F.log_softmax(x, dim=1)
```

### Model Summary

<p align="left">    
    <img width="600" aling="right" src="https://github.com/Paurnima-Chavan/mnist-S6/assets/25608455/a72b1e11-a4e2-4ebe-9e75-4929c4069e48" />
</p>

We will be building the following network, as you can see it We will be building the following network, as you can see it contains **seven Convolution layers**, **two Max pooling layers** (to reduce channel size into half), **two 1*1 layers** followed by **Avg pooling layer.**

## Usage

To use the model for handwritten digit recognition, followed below steps:

- Instantiate an instance of the Net class.
- Load the MNIST dataset and preprocess it as required.    
- Pass the preprocessed input through the model's forward method to obtain the predicted digit class probabilities.
- We have used SGD (Stochastic Gradient Descent) optimizer, which will be used to update the model's parameters during training. The learning rate is set to 0.01, and the momentum is set to 0.9
- Trained the model for the specified number of epochs, printing the epoch number, and performs training and testing steps in each epoch. The learning rate is adjusted using the scheduler, allowing better optimization over time. The train and test functions are responsible for the actual training and testing processes, respectively.   
- Finally, plotted the training and testing accuracy as well as the training and testing loss. It creates a 2x2 grid of subplots in a figure to visualize the training and testing performance over epochs, providing insights into the model's learning progress.

<p align="center">    
    <img width="800" hight="300" aling="right" src="https://github.com/Paurnima-Chavan/mnist-S6/assets/25608455/59500e1e-d1ec-46b2-85cc-1762d6b0936c" />
 </p>

## Summary
In summary, our efforts have yielded a successfully established environment using PyTorch and TorchVision. Leveraging this environment, we were able to classify handwritten digits from the MNIST dataset with remarkable accuracy, achieving a 99.4% accuracy rate using only 16.5 k parameters.
