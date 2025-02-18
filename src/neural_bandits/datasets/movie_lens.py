import logging
import os
from typing import Tuple

import pandas as pd
import torch

from neural_bandits.datasets.abstract_dataset import AbstractDataset

import urllib
import zipfile


logger = logging.getLogger(__name__)


def _download_movielens(dest_path: str = "./data") -> None:
    """Downloads the 'Small' MovieLens dataset if it does not already exist. See
    (from https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)  for further information.
    """
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

    zip_file = os.path.join(dest_path, "ml-latest-small.zip")
    if not os.path.exists(zip_file):
        logger.info("Downloading dataset...")
        urllib.request.urlretrieve(url, zip_file)
        logger.info("Download completed.")
    else:
        logger.info("Dataset already downloaded.")


def _extract_movielens(zip_path: str, extract_dir: str) -> None:
    """Extract the MovieLens dataset archive if `extract_dir` does not have a directory called ml-latest-small."""

    if not os.path.exists(os.path.join(extract_dir, "ml-latest-small")):
        logger.info("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
    else:
        logger.info("Could not extract dataset; directory already exists.")


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


def _setup_movielens(dest_path: str = "./data", k: int = 4, L: int = 200, min_movies = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Download, extract, and load the MovieLens dataset."""
    
    # Check if features cached in `dest_path`. If not download and calculate.
    if os.path.exists(os.path.join(dest_path, "ml-latest-small", "movielens_small_user_features.pt")):
        user_features = torch.load(os.path.join(dest_path, "ml-latest-small", "movielens_small_user_features.pt"))
        movie_features = torch.load(os.path.join(dest_path, "ml-latest-small", "movielens_small_movie_features.pt"))
        history = torch.load(os.path.join(dest_path, "ml-latest-small", "movielens_small_history.pt"))
        future = torch.load(os.path.join(dest_path, "ml-latest-small", "movielens_small_future.pt"))
        return user_features, movie_features, history, future
    else:
        _download_movielens(dest_path)
        _extract_movielens(os.path.join(dest_path, "ml-latest-small.zip"), dest_path)
        data = _load_movielens_data(os.path.join(dest_path, "ml-latest-small"))
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
        
        # `data` now has columns `userId` (1->|Users|), `movieId` (1->|Movies|), `rating` (1-5), `timestamp`
        # We will only use `userId` and `movieId` for now (inspired by the approach from Li et. al., 2010 (see 
        # https://arxiv.org/abs/1003.0146))
        
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
        
        print(user_features.shape)
        print(movie_features.shape)
        
        # Store the features, history and future.
        torch.save(user_features, os.path.join(dest_path, "ml-latest-small", "movielens_small_user_features.pt"))
        torch.save(movie_features, os.path.join(dest_path, "ml-latest-small", "movielens_small_movie_features.pt"))
        torch.save(history, os.path.join(dest_path, "ml-latest-small", "movielens_small_history.pt"))
        torch.save(future, os.path.join(dest_path, "ml-latest-small", "movielens_small_future.pt"))
        
        return user_features, movie_features, history, future
    


class MovieLensDataset(AbstractDataset[torch.Tensor]):
    """
    Args:
        csv_file: Path to a CSV file with MovieLens watch data.
                        The CSV should have at least columns: 'user_id', 'movie_id'.
        p: The probability that a watch event (A(i,j)=1) is placed in the history H.
        num_movies: Number of movies to sample (the candidate set will be of this size).
        num_steps: How many time steps (samples) the dataset will simulate.
        svd_rank: Rank (number of latent dimensions) for the SVD decomposition.
        random_state: Seed for reproducibility.
    """
    
    num_actions: int = 87585 # Number of Movies.
    context_size: int = 20 * 20 # Number of Latent Dimensions squared. The outer product of the user and movie features.
    num_samples: int = 200948 # Number of Users.
    
    def __init__(
        self, dest_path="./data", svd_rank=20, outer_product=True, k=4, L=200
    ):

        super().__init__(needs_disjoint_contextualization=False)
        self.user_features, self.movie_features, self.history, self.F = _setup_movielens(
            dest_path=dest_path, k=k, L=L
        )
        
        # Contexts per User:
        self.contextualized_actions_per_user: torch.Tensor
        if outer_product:
            self.contextualized_actions_per_user = torch.einsum("ui,mj->umij", self.user_features, self.movie_features).flatten(start_dim=2)
        else:
            self.contextualized_actions_per_user = torch.cat(
                (self.user_features.unsqueeze(1).expand(-1, self.movie_features.size(0), -1), self.movie_features.unsqueeze(0).expand(self.user_features.size(0), -1, -1)),
                dim=-1
            ) # Shape: (num_users, num_movies, 2 * svd_rank)
            self.context_size = 2 * svd_rank
        

    def __len__(self) -> int:
        return self.user_features.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get avaiable actions (1 - history[userId - 1 = idx])
        available_actions = (1.0 - self.history[idx]).bool()
        
        # Get the context for each action
        contexts = self.contextualized_actions_per_user[idx][available_actions]
        
        return contexts, torch.tensor([self.reward(idx, movie_idx) for movie_idx in range(self.history.shape[-1]) if available_actions[movie_idx]], dtype=torch.float32)
        

    def reward(self, idx: int, action: int) -> float:
        # An idx represents a user and the action is a movie.
        return self.F[idx, action].item()
