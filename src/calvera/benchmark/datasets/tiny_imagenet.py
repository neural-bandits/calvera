import logging
import os
import urllib.request
import zipfile
from typing import Literal, cast

import torch
import torchvision

from calvera.benchmark.datasets.abstract_dataset import AbstractDataset

logger = logging.getLogger(__name__)


def _download_tinyimagenet(dest_path: str = "./data") -> None:
    """Downloads the TinyImageNet dataset if it does not already exist.

    Args:
        dest_path: The directory where the dataset will be stored.
    """
    dataset_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    file_name = "tiny-imagenet-200.zip"

    zip_file = os.path.join(dest_path, file_name)
    if not os.path.exists(zip_file):
        logger.info("Downloading TinyImageNet dataset...")
        urllib.request.urlretrieve(dataset_url, zip_file)
        logger.info("Download completed.")
    else:
        logger.info("Dataset already downloaded.")


def _extract_tinyimagenet(zip_path: str, extract_dir: str) -> None:
    """Extract the TinyImageNet dataset archive to the specified directory."""
    assert os.path.exists(zip_path), f"Could not find zip file at {zip_path}"

    logger.info("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    os.remove(zip_path)
    logger.info("Extraction completed.")


def _restructure_val_folder(dataset_folder: str) -> None:
    """Restructure validation folder to match ImageFolder expected format."""
    val_dir = os.path.join(dataset_folder, "val")
    val_img_dir = os.path.join(val_dir, "images")

    if any(os.path.isdir(os.path.join(val_img_dir, d)) for d in os.listdir(val_img_dir)):
        return

    val_annotations_path = os.path.join(val_dir, "val_annotations.txt")

    val_img_dict = {}
    with open(val_annotations_path) as f:
        for line in f.readlines():
            parts = line.strip().split("\t")
            val_img_dict[parts[0]] = parts[1]

    for class_name in set(val_img_dict.values()):
        os.makedirs(os.path.join(val_img_dir, class_name), exist_ok=True)

    for img_name, class_name in val_img_dict.items():
        os.rename(os.path.join(val_img_dir, img_name), os.path.join(val_img_dir, class_name, img_name))

    logger.info("Validation folder restructured for ImageFolder compatibility.")


def _restructure_test_folder(dataset_folder: str) -> None:
    """Restructure test folder to work with ImageFolder."""
    test_dir = os.path.join(dataset_folder, "test", "images")
    unknown_dir = os.path.join(test_dir, "unknown")
    os.makedirs(unknown_dir, exist_ok=True)

    img_files = [f for f in os.listdir(test_dir) if f.endswith(".JPEG") and os.path.isfile(os.path.join(test_dir, f))]

    for img_file in img_files:
        os.rename(os.path.join(test_dir, img_file), os.path.join(unknown_dir, img_file))

    logger.info("Test folder restructured for ImageFolder compatibility.")


def _setup_tinyimagenet(
    dest_path: str = "./data",
    split: str = "train",
) -> tuple[torchvision.datasets.folder.ImageFolder, torch.Tensor]:
    """Download, extract, and set up the TinyImageNet dataset.

    Args:
        dest_path: The directory where the dataset will be stored.
        split: Which split to use ('train', 'val', or 'test')

    Returns:
        image_dataset: The ImageFolder dataset for the specified split
        labels: Tensor of labels corresponding to the dataset
    """
    dataset_folder = os.path.join(dest_path, "tiny-imagenet-200")

    if not os.path.exists(dataset_folder):
        _download_tinyimagenet(dest_path)
        _extract_tinyimagenet(os.path.join(dest_path, "tiny-imagenet-200.zip"), dest_path)

    if split == "val":
        _restructure_val_folder(dataset_folder)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    if split == "train":
        folder_path = os.path.join(dataset_folder, "train")
    elif split == "val":
        folder_path = os.path.join(dataset_folder, "val", "images")
    else:  # test
        folder_path = os.path.join(dataset_folder, "test", "images")
        if not os.path.exists(os.path.join(folder_path, "unknown")):
            _restructure_test_folder(dataset_folder)

    try:
        image_dataset = torchvision.datasets.folder.ImageFolder(folder_path, transform=transform)
    except (RuntimeError, FileNotFoundError) as e:
        raise RuntimeError(
            f"Error loading dataset from {folder_path}: {e}. "
            f"Make sure the dataset is downloaded and correctly structured."
        ) from e

    labels = torch.tensor([label for _, label in image_dataset.samples], dtype=torch.int64)

    return image_dataset, labels


class TinyImageNetDataset(AbstractDataset[torch.Tensor]):
    """Loads the Tiny ImageNet dataset as a PyTorch Dataset.

    More information can be found at https://cs231n.stanford.edu/reports/2015/pdfs/yle_project.pdf.

    Tiny ImageNet has 200 classes with 500 training images, 50 validation images,
    and 50 test images per class. Each image is 64x64 pixels in 3 channels (RGB).
    """

    num_actions: int = 200
    context_size: int = 3 * 64 * 64
    num_samples: int = 100000

    def __init__(
        self,
        dest_path: str = "./data",
        split: Literal["train", "val", "test"] = "train",
        max_classes: int = 10,
    ) -> None:
        """Initialize the Tiny ImageNet dataset.

        Args:
            dest_path: The directory where the dataset will be stored.
            split: Which split to use ('train', 'val', or 'test')
            max_classes: The maximum number of classes to use from the dataset. Default is 200.
        """
        super().__init__(needs_disjoint_contextualization=False)

        self.dest_path = dest_path
        self.split = split

        self.image_dataset, self.y = _setup_tinyimagenet(
            dest_path=dest_path,
            split=split,
        )

        if max_classes < 200:
            self.image_dataset.classes = self.image_dataset.classes[:max_classes]
            self.image_dataset.class_to_idx = {c: i for i, c in enumerate(self.image_dataset.classes)}
            # filter out samples from classes not in the first `max_classes`
            self.image_dataset.samples = [
                (path, label) for path, label in self.image_dataset.samples if label < max_classes
            ]
            self.y = self.y[self.y < max_classes]
            self.num_actions = max_classes

        self.idx_to_class = {v: k for k, v in self.image_dataset.class_to_idx.items()}

        self.X = None

    def _lazy_load_X(self) -> torch.Tensor:
        """Lazily load the images into a tensor.

        Returns:
            The flattened image tensor with shape (num_samples, 3*64*64)
        """
        if self.X is None:
            logger.info("Loading all images into memory...")
            X_list = []
            for idx in range(len(self.image_dataset)):
                img, _ = self.image_dataset[idx]
                img_flat = img.view(-1)
                X_list.append(img_flat)
            self.X = torch.stack(X_list)  # type: ignore
        return cast(torch.Tensor, self.X)

    def __len__(self) -> int:
        """Return the number of contexts/samples in this dataset."""
        return len(self.image_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the context (flattened image) and rewards for a given index.

        Args:
            idx: The index of the context in this dataset.

        Returns:
            context: The flattened image features as the context.
            rewards: The rewards for each action (1.0 for correct class, 0.0 otherwise).
        """
        image_tensor, _ = self.image_dataset[idx]

        # Flatten the image tensor from (C, H, W) to (C*H*W)
        context = image_tensor.view(1, -1)  # shape: (1, 3*64*64)

        rewards = torch.tensor(
            [self.reward(idx, action) for action in range(self.num_actions)],
            dtype=torch.float32,
        )

        return context, rewards

    def reward(self, idx: int, action: int) -> float:
        """Return the reward for a given index and action.

        1.0 if the action is the same as the label, 0.0 otherwise.

        Args:
            idx: The index of the context in this dataset.
            action: The action for which the reward is requested.
        """
        return float(self.y[idx] == action)

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        return f"{self.__class__.__name__}(split={self.split}, num_samples={len(self)})"
