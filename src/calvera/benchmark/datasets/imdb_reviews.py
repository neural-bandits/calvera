import logging
import os
import pathlib
import re
import tarfile
import urllib.request
from typing import Literal, cast

import pandas as pd
import torch
from transformers import BertTokenizer, DataCollatorForTokenClassification, PreTrainedTokenizer

from calvera.benchmark.datasets.abstract_dataset import AbstractDataset

logger = logging.getLogger(__name__)


def _download_imdb_data(dest_path: str) -> None:
    """Download the IMDB dataset archive if it does not already exist.

    More information can be found at https://ai.stanford.edu/~amaas/data/sentiment/.
    """
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    tar_file = os.path.join(dest_path, "aclImdb_v1.tar.gz")
    if not os.path.exists(tar_file):
        logger.info("Downloading dataset...")
        urllib.request.urlretrieve(url, tar_file)
        logger.info("Download completed.")
    else:
        logger.info("Dataset already downloaded.")


def _extract_data(tar_path: str, extract_dir: str) -> None:
    """Extract the tar.gz dataset archive."""
    extracted_folder = os.path.join(extract_dir, "aclImdb")
    if not os.path.exists(extracted_folder):
        logger.info("Extracting dataset...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        logger.info("Extraction completed.")
    else:
        logger.info("Dataset already extracted.")


def _load_imdb_data(data_dir: str, subset: str = "train") -> tuple[list[str], list[int]]:
    """Load IMDB reviews and labels from the specified subset directory.

    Assumes a directory structure: aclImdb/{train,test}/{pos,neg}
    """
    texts = []
    labels = []
    for label_type in ["pos", "neg"]:
        dir_path = os.path.join(data_dir, subset, label_type)
        for filename in os.listdir(dir_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(dir_path, filename)
                with open(file_path, encoding="utf-8") as f:
                    texts.append(f.read())
                # Label as 1 for positive reviews, 0 for negative reviews
                labels.append(1 if label_type == "pos" else 0)
    return texts, labels


def _setup_dataset(
    partition: Literal["train", "test"] = "train",
    dest_path: str | None = None,
) -> pd.DataFrame:
    """Download and setup the dataset."""
    dest_path_or_current_path = (
        dest_path if dest_path is not None else os.path.join(pathlib.Path(__file__).parent.absolute())
    )
    if os.path.exists(dest_path_or_current_path) and not os.path.exists(
        os.path.join(dest_path_or_current_path, "aclImdb")
    ):
        _download_imdb_data(dest_path_or_current_path)
        _extract_data(
            os.path.join(dest_path_or_current_path, "aclImdb_v1.tar.gz"),
            dest_path_or_current_path,
        )

    texts, sentiments = _load_imdb_data(os.path.join(dest_path_or_current_path, "aclImdb"), partition)

    data = pd.DataFrame({"text": texts, "sentiment": sentiments})  # 1 for positive, 0 for negative

    data.drop_duplicates(inplace=True)
    data["text"] = data["text"].apply(_preprocess_text)
    data = data.reset_index(drop=True)

    return data


def _preprocess_text(text: str) -> str:
    """Preprocess the text by removing special characters, removing urls and lowercasing it."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text


# Type for the input to the transformer model.
# The input is a tuple containing the `input_ids`, `attention_mask`, and `token_type_ids`.
TextActionInputType = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class ImdbMovieReviews(AbstractDataset[TextActionInputType]):
    """A dataset for the IMDB movie reviews sentiment classification task.

    More information can be found at https://ai.stanford.edu/~amaas/data/sentiment/.

    Args:
        dest_path: The path to the directory where the dataset is stored. If None, the dataset will be downloaded to
            the current directory.
        partition: The partition of the dataset to use. Either "train" or "test".
        max_len: The maximum length of the input text. If the text is longer than this, it will be truncated. If it is
            shorter, it will be padded. This is also the `context_size` of the dataset.
        tokenizer: A tokenizer from the `transformers` library. If None, the `BertTokenizer` will be used.
    """

    num_actions: int = 2  # 1 for positive, 0 for negative
    # We cannot provide a context size directly since the context is the text itself. You should use the output of this
    # dataset as the input to a transformer model. Then you can use the output of the model as the context, then apply
    # the `MultiClassContextualizer` to it.
    context_size: int = 256
    # We only provide the number of samples for the training set here.
    num_samples: int = 24904

    def __init__(
        self,
        dest_path: str = "./data",
        partition: Literal["train", "test"] = "train",
        max_len: int = 255,
        tokenizer: PreTrainedTokenizer | None = None,
    ):
        """Initialize the IMDB movie reviews dataset.

        Args:
            dest_path: The path to the directory where the dataset is stored. If None, the dataset will be downloaded
                to the current directory.
            partition: The partition of the dataset to use. Either "train" or "test".
            max_len: The maximum length of the input text. If the text is longer than this, it will be truncated.
            tokenizer: A tokenizer from the `transformers` library. If None, the `BertTokenizer` will be used.
        """
        # Using disjoint contextualization for this dataset does not work. We have a sequence of tokens.
        super().__init__(needs_disjoint_contextualization=False)

        self.data = _setup_dataset(
            partition=partition,
            dest_path=dest_path,
        )

        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", padding="max_length", truncation=True)

    def __len__(self) -> int:
        """Return the number of samples in this dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[TextActionInputType, torch.Tensor]:
        """Return the input and reward for the given index.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A tuple containing the necessary input for a model from the `transformers` library and the reward.
            Specifically, the input is a tuple containing the `input_ids`, `attention_mask`, and `token_type_ids`.
            (cmp. https://huggingface.co/docs/transformers/v4.49.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__)
        """
        inputs = self.tokenizer(
            self.data["text"][idx],
            None,
            add_special_tokens=True,
            max_length=self.context_size,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
        )

        rewards = torch.tensor(
            [self.reward(idx, action) for action in range(self.num_actions)],
            dtype=torch.float,
        )

        return (
            (
                torch.tensor(inputs["input_ids"], dtype=torch.long).unsqueeze(0),
                torch.tensor(inputs["attention_mask"], dtype=torch.long).unsqueeze(0),
                torch.tensor(inputs["token_type_ids"], dtype=torch.long).unsqueeze(0),
            ),
            rewards,
        )

    def reward(self, idx: int, action: int) -> float:
        """Return the reward for the given index and action.

        1.0 if the action is the correct sentiment, 0.0 otherwise.

        Args:
            idx: The index of the sample.
            action: The action to evaluate.
        """
        return 1.0 if action == self.data["sentiment"][idx] else 0.0

    def get_data_collator(self, padding: bool | str = True) -> DataCollatorForTokenClassification:
        """Return a data collator for token classification tasks.

        Args:
            padding: Either a boolean or a string. If True, the data collator will pad the inputs. If False, it will
                not pad the inputs. If a string, it will use the string as the padding token. Default is True.

        Returns:
            A data collator for token classification tasks.
        """
        return DataCollatorForTokenClassification(tokenizer=self.tokenizer, padding=padding)

    def sort_key(self, idx: int) -> int:
        """Return the label for a given index."""
        return cast(int, self.data["sentiment"][idx])
