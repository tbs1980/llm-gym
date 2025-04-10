import logging

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


mean_activity_of_pixels = None

def SubtractMeanPixelActivity(x: np.array):
    """
    A method that subracts precomputed mean activity of pixels from each pixel.
    The mean activity of pixels are computed using the entire training set.

    Parameters
    ----------

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
        shuffle=False,
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
            transforms.CenterCrop(227),
        ]
    )

    trainset = torchvision.datasets.ImageNet(
        root=train_dataset_path, split="train", transform=preprocess
    )
    logging.debug("Train Dataset created")

    train_loader = torch.utils.data.DataLoader(
        trainset,
        shuffle=False,
        num_workers=1,
        drop_last=True,
        batch_size=256,
        persistent_workers=True,
    )
    logging.debug("Train Dataloader created")

    return train_loader


if __name__ == "__main__":

    train_dataset_path = "/home/sree/data/ILSVRC2012"
    compute_mean_activity = False

    if compute_mean_activity is True:
        mean_pixels = compute_mean_pixels(train_dataset_path)

        # assign the global variable defined at the top
        mean_activity_of_pixels = mean_pixels.numpy()
        np.save("mean_activity_of_pixels.npy", mean_activity_of_pixels)
    else:
        mean_activity_of_pixels = np.float32(np.load('/home/sree/github/llm-gym/foundations/2012_alexnet/mean_activity_of_pixels.npy'))
        logging.debug(f"Shape of the mean activity of pixels loaded is {mean_activity_of_pixels.shape}")

    train_data_loader = get_train_data_loader(train_dataset_path)

    dataiter = iter(train_data_loader)
    images, labels = next(train_data_loader)


    
