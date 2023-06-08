import torch.utils.data
from torchvision import datasets, transforms


def load_minst_data(batch_size):
    """
   This function is designed to retrieve the MNIST dataset from the PyTorch library, apply several transformations
   to it, and ultimately provide data loaders for both the training and testing sets.
    :param batch_size: Batch size refers to the number of samples or data points that are processed simultaneously
                       or at a given time during the training or inference process.
    :return: Data loader object for both the training and testing sets
    """

    # Train data transformations
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
        transforms.Resize((28, 28)),
        transforms.RandomRotation((-15., 15.), fill=0),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # mean and SD (we need to calculate) if RGB 6 values are here
    ])

    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # keep same as train
    ])

    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('../data', train=False, download=False, transform=test_transforms)

    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

    return train_loader, test_loader
