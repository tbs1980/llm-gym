import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

mean_activity_of_pixels = None


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1000),
        )
        self.init_weights_and_biases()

    def init_weights_and_biases(self):
        logging.debug("Initialising weights and biases...")

        for layer in self.model:
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
        # original paper = 1 for Conv2d layers 2nd, 4th, and 5th conv layers
        nn.init.constant_(self.model[4].bias, 1)
        nn.init.constant_(self.model[10].bias, 1)
        nn.init.constant_(self.model[12].bias, 1)


def SubtractMeanPixelActivity(x: torch.Tensor):
    """
    A method that subracts precomputed mean activity of pixels from each pixel.
    The mean activity of pixels are computed using the entire training set.

    Parameters
    ----------
    x: torch.Tensor
        An input image as torch tensor in C x H x W shape.
    """
    return x - mean_activity_of_pixels


def ToTensorNoScaling(x: np.array) -> torch.Tensor:
    """
    A method for coverting RGB image in numpy format into torch Tensor without 0-1 scaling.
    Also convert the image into C x H x W format where C is the number of channels
    H is the height and W is the width of the image.

    Parameters
    ----------
    x: np.array
        The input image (numpy array)

    Returns
    -------
    Corresponding torch tensor in C x H x W shape
    """

    return torch.from_numpy(np.array(x).transpose(2, 0, 1))


def compute_mean_pixels(train_dataset_path: str) -> np.array:
    """
    A method that calculates the mean pixel values for the entire ILCSVR2012 training set.
    We assume that the follwing preprocessing will be performed on the raw image
        1. A resise of the image into 256 pixels
        2. A center crop of the image with 227x227 pixels
        3. Converr the image into C x H x W format

    If the image gets processed in other ways, these should be reflected in the above
    set of operations before computing the mean activity of pixels. For example, if the
    image gets pricessed with transforms.ToTensor() method from PyTorch, then it gets
    scaled into 0-1 range automatically. Therefore, the mean activity will be different.

    Parameters
    ----------
    train_dataset_path: str
        The path where the train dataset can be found.

    Returns
    -------
    Mean activity of pixels in the tranining dataset in C x H x W shape.

    """
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(227),
            transforms.Lambda(ToTensorNoScaling),
        ]
    )

    trainset = torchvision.datasets.ImageNet(
        root=train_dataset_path, split="train", transform=preprocess
    )
    logging.debug("Train Dataset created")

    train_data_loader = torch.utils.data.DataLoader(
        trainset,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        batch_size=256,
        persistent_workers=True,
    )
    logging.debug("Train Dataloader created")

    sum_pixels = []
    sum_imgs = 0
    for imgs, _ in tqdm(train_data_loader, desc="Comuting sum of pixels"):
        sum_pixels.append(imgs.sum(dim=0))
        sum_imgs += imgs.shape[0]

    mean_pixels = torch.stack(sum_pixels).sum(dim=0) / sum_imgs

    return mean_pixels


def get_train_data_loader(train_dataset_path: str) -> torch.utils.data.DataLoader:
    """
    A function for getting train data loader with the following preprocessing
    operations:
        1. Resize the image to 256 pixels
        2. CenterCrop to 227 x 227 pixels
        3. Covert to torch.Tensor with shape C x H x W shape
        4. Subtract the mean activity of pixels from each pixels for each channel

    Parameters
    ----------
    train_dataset_path: str
        The path where the train dataset can be found.

    Returns
    -------
    A instance of the train data loader.
    """
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(227),
            transforms.Lambda(ToTensorNoScaling),
            transforms.Lambda(SubtractMeanPixelActivity),
        ]
    )

    trainset = torchvision.datasets.ImageNet(
        root=train_dataset_path, split="train", transform=preprocess
    )
    logging.debug("Train Dataset created")

    train_loader = torch.utils.data.DataLoader(
        trainset,
        shuffle=False,
        num_workers=4,
        drop_last=True,
        batch_size=128,
        persistent_workers=True,
    )
    logging.debug("Train Dataloader created")

    return train_loader


def get_valid_data_loader(valid_dataset_path: str) -> torch.utils.data.DataLoader:
    """
    A function for getting train data loader with the following preprocessing
    operations:
        1. Resize the image to 256 pixels
        2. CenterCrop to 227 x 227 pixels
        3. Covert to torch.Tensor with shape C x H x W shape
        4. Subtract the mean activity of pixels from each pixels for each channel

    Parameters
    ----------
    valid_dataset_path: str
        The path where the validation dataset can be found.

    Returns
    -------
    A instance of the validation data loader.
    """
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(227),
            transforms.Lambda(ToTensorNoScaling),
            transforms.Lambda(SubtractMeanPixelActivity),
        ]
    )

    validset = torchvision.datasets.ImageNet(
        root=valid_dataset_path, split="val", transform=preprocess
    )
    logging.debug("Validation Dataset created")

    valid_loader = torch.utils.data.DataLoader(
        validset,
        shuffle=False,
        num_workers=2,
        drop_last=True,
        batch_size=256,
        persistent_workers=True,
    )
    logging.debug("Validation Dataloader created")

    return valid_loader


if __name__ == "__main__":
    logging.basicConfig(filename="myapp.log", level=logging.INFO)

    dataset_path = "/home/sree/data/ILSVRC2012"
    compute_mean_activity = False

    if compute_mean_activity is True:
        mean_pixels = compute_mean_pixels(dataset_path)

        # assign the global variable defined at the top
        mean_activity_of_pixels = mean_pixels
        np.save("mean_activity_of_pixels.npy", mean_activity_of_pixels.numpy())
    else:
        mean_activity_of_pixels = torch.Tensor(
            np.float32(
                np.load(
                    "/home/sree/github/llm-gym/foundations/2012_alexnet/mean_activity_of_pixels.npy"
                )
            )
        )
        logging.debug(
            f"Shape of the mean activity of pixels loaded is {mean_activity_of_pixels.shape}"
        )

    train_data_loader = get_train_data_loader(dataset_path)
    valid_data_loader = get_valid_data_loader(dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.debug(
        f"Using device = {device}",
    )

    alexnet = AlexNet().to(device)
    logging.debug("AlexNet initiated")

    optimizer = optim.Adam(params=alexnet.parameters(), lr=0.0001)
    logging.debug("Optimizer initiated")

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    logging.debug("Learning rate scheduler initiated")

    loss = torch.nn.CrossEntropyLoss()
    logging.debug("Cross entropy loss function created")
