import logging
import os
from typing import Literal, Tuple

import pandas as pd
import torch

from neural_bandits.datasets.abstract_dataset import AbstractDataset

import urllib
import zipfile


logger = logging.getLogger(__name__)


def _download_movielens(dest_path: str = "./data", version: Literal["ml-32m", "ml-latest-small"] = "ml-latest-small") -> None:
    """Downloads the 'small' MovieLens dataset if it does not already exist. See
    (from https://files.grouplens.org/datasets/movielens)  for further information.
    """
    file_name = f"{version}.zip"
    url = "https://files.grouplens.org/datasets/movielens/" + file_name

    zip_file = os.path.join(dest_path, file_name)
    if not os.path.exists(zip_file):
        logger.info("Downloading dataset...")
        urllib.request.urlretrieve(url, zip_file)
        logger.info("Download completed.")
    else:
        logger.info("Dataset already downloaded.")


def _extract_movielens(zip_path: str, extract_dir: str) -> None:
    """Extract the MovieLens dataset archive if `extract_dir` exists."""
    assert os.path.exists(zip_path), f"Could not find zip file at {zip_path}"
    
    logger.info("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def _load_movielens_data(data_dir: str) -> pd.DataFrame:
    """Load the MovieLens ratings data from the specified directory."""
    ratings_path = os.path.join(data_dir, "ratings.csv")
    return pd.read_csv(ratings_path)


def _build_movielens_features(history: torch.Tensor, svd_rank: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the user and movie features for the MovieLens dataset."""
    U_full, S_full, Vt_full = torch.linalg.svd(history, full_matrices=False)
    U_r = U_full[:, :svd_rank]  # shape: (num_users, svd_rank)
    S_r = S_full[:svd_rank]  # shape: (svd_rank,)
    Vt_r = Vt_full[:svd_rank, :]  # shape: (svd_rank, num_movies)

    # We write the decomposition as H = U_r * diag(S_r) * Vt_r.
    # To match the formulation H = U S M^T, one common choice is to absorb the singular values
    # into the user features. That is, we define:
    #    user_features = U_r * diag(S_r)   (so that each u_i is a vector of length svd_rank)
    #    movie_features = Vt_r^T           (so that each m_j is a vector of length svd_rank)
    U_features = U_r * S_r  # broadcasting S_r over the columns
    
    return U_features, Vt_r.T


def _setup_movielens(dest_path: str = "./data", k: int = 4, L: int = 200, min_movies = 10, store_features: bool = True, version: Literal["ml-latest-small", "ml-32m"] = "ml-latest-small") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Download, extract, and load the MovieLens dataset
    
    Args:
        dest_path: The directory where the dataset will be stored.
        k: The number of movies to exclude per user.
        L: The number of movies to include in the dataset.
        min_movies: The minimum number of movies a user must have rated to be included in the dataset (after only 
            taking the top `L` movies).
        store_features: Whether to store the user and movie features.
        version: The version of the MovieLens dataset to use. Either "small" or "32m".
        
    Returns:
        user_features: The user features.
        movie_features: The movie features.
        history: The history matrix.
        future: The future matrix.
    """
    file_postfix = f"_k{k}_L{L}_min{min_movies}"
    
    
    # Check if features cached in `dest_path`. If not download and calculate.
    if os.path.exists(os.path.join(dest_path, version, f"user_features{file_postfix}.pt")):
        user_features = torch.load(os.path.join(dest_path, version, f"user_features{file_postfix}.pt"))
        movie_features = torch.load(os.path.join(dest_path, version, f"movie_features{file_postfix}.pt"))
        history = torch.load(os.path.join(dest_path, version, f"history{file_postfix}.pt"))
        future = torch.load(os.path.join(dest_path, version, f"future{file_postfix}.pt"))
        return user_features, movie_features, history, future
    else:
        _download_movielens(dest_path, version)
        _extract_movielens(os.path.join(dest_path, version + ".zip"), dest_path)
        data = _load_movielens_data(os.path.join(dest_path, version))
        data = data.dropna()
        data = data.reset_index(drop=True)
        
        # Only keep the L most common movies.
        movie_counts = data["movieId"].value_counts()
        top_L_movies = movie_counts[:L].index
        data = data[data["movieId"].isin(top_L_movies)]
        
        # Only keep users that have rated at least `min_movies` movies.
        user_counts = data["userId"].value_counts()
        top_users = user_counts[user_counts >= min_movies].index
        data = data[data["userId"].isin(top_users)]
        
        data = data.reset_index(drop=True)
        
        # Convert user and movie ids to integers.
        data["userId"] = data["userId"].astype("int")
        data["movieId"] = data["movieId"].astype("int")
        
        # We will only use `userId` and `movieId` for now (inspired by the approach from Li et. al., 2010 (see 
        # https://arxiv.org/abs/1003.0146))
        # Additionally, we will use the `timestamp` to split the data into history and future.
        
        # Build the complete `viewed` relationship matrix.
        has_viewed = torch.zeros((data["userId"].nunique(), data["movieId"].nunique()), dtype=torch.float32)
        
        movie_id_to_index = {movie_id: i for i, movie_id in enumerate(data["movieId"].unique())}
        user_id_to_index = {user_id: i for i, user_id in enumerate(data["userId"].unique())}
        
        kthlast_timestamp_per_user = data.groupby("userId", group_keys=False)[["userId", "movieId", "timestamp"]].apply(lambda x: x.nlargest(k + 1, columns="timestamp")).groupby("userId")[["userId", "timestamp"]].min()
        # Add the last |movies_rated| - k movies to the history per user.
        history = has_viewed.clone()
        future = torch.zeros_like(history)
        for _, row in data.iterrows():
            user_id = row["userId"].item()
            kth_timestamp = kthlast_timestamp_per_user.loc[user_id, "timestamp"]
            if row["timestamp"].item() > kth_timestamp.item():
                future[user_id_to_index[user_id], movie_id_to_index[row["movieId"].item()]] = 1 
        history = history - future
        
        user_features, movie_features = _build_movielens_features(history=history)
        
        # Store the features, history and future.
        if store_features:
            torch.save(user_features, os.path.join(dest_path, version, f"user_features{file_postfix}.pt"))
            torch.save(movie_features, os.path.join(dest_path, version, f"movie_features{file_postfix}.pt"))
            torch.save(history, os.path.join(dest_path, version, f"history{file_postfix}.pt"))
            torch.save(future, os.path.join(dest_path, version, f"future{file_postfix}.pt"))
        
        return user_features, movie_features, history, future
    


class MovieLensDataset(AbstractDataset[torch.Tensor]):
    """MovieLens dataset for combinatorial contextual bandits. We build the context by using the SVD decomposition of the
    user-movie matrix. The context is the outer product of the user and movie features.
    
    Args:
        dest_path: The directory where the dataset is / will be stored.
        svd_rank: Rank (number of latent dimensions) for the SVD decomposition.
        outer_product: Whether to use the outer product of the user and movie features as the context. If False, the
            context will be the concatenation of the user and movie features. (Might perform better for Neural Bandits).
        k: The number of movies to exclude per user.
        L: The number of movies to include in the dataset. (Top L most common movies).
    """
    
    num_actions: int = -1 # There is no constant number of actions in the MovieLens dataset.
    num_samples: int = -1 
    context_size: int = -1
    
    def __init__(
        self, dest_path="./data", svd_rank=20, outer_product=True, k=4, L=200, version: Literal["ml-latest-small", "ml-32m"] = "ml-latest-small"
    ):
        super().__init__(needs_disjoint_contextualization=False)
        self.user_features, self.movie_features, self.history, self.F = _setup_movielens(
            dest_path=dest_path, k=k, L=L, version=version
        )
        
        self.outer_product = outer_product
        
        # We can predict k movies per user. The idea is that we only predict a user once.
        self.num_actions = self.history.shape[-1]
        

    def __len__(self) -> int:
        return self.user_features.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get avaiable actions (1 - history[userId - 1 = idx])
        available_actions = (1.0 - self.history[idx]).bool()
        
        # Get the context for each action
        contexts: torch.Tensor
        
        if self.outer_product:
            contexts = torch.einsum("ij,jk->ijk", self.user_features[idx], self.movie_features)
        else:
            contexts = torch.cat(
                (self.user_features[idx].unsqueeze(0).expand(self.movie_features.size(0), -1), self.movie_features),
                dim=-1
            )
        
        return contexts, torch.tensor([self.reward(idx, movie_idx) for movie_idx in range(self.history.shape[-1]) if available_actions[movie_idx]], dtype=torch.float32)
        

    def reward(self, idx: int, action: int) -> float:
        # An idx represents a user and the action is a movie.
        return self.F[idx, action].item()
