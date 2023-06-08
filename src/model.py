import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


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

        self.fc1 = nn.Sequential(
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
        x = self.fc1(x)
        x = x.squeeze()
        return F.log_softmax(x, dim=1)


def model_summary(model, input_size):
    """
    This function displays a summary of the model, providing information about its architecture,
    layer configuration, and the number of parameters it contains.
    :param model: model
    :param input_size: input_size for model
    :return:
    """
    summary(model, input_size=input_size)



