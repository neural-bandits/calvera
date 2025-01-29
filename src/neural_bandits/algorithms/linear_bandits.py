import torch

from .abstract_bandit import AbstractBandit


class LinearBandit(AbstractBandit):
    def __init__(self, n_features: int) -> None:
        super().__init__(n_features)

        self.precision_matrix: torch.Tensor = torch.eye(n_features)
        self.b = torch.zeros(n_features)
        self.theta = torch.zeros(n_features)


class LinearTSBandit(LinearBandit):
    def __init__(self, n_features: int) -> None:
        super().__init__(n_features)

    def forward(self, contextualised_actions: torch.Tensor) -> torch.Tensor:
        assert (
            contextualised_actions.shape[2] == self.n_features
        ), "Contextualised actions must have shape (batch_size, n_arms, n_features)"
        batch_size = contextualised_actions.shape[0]
        n_arms = contextualised_actions.shape[1]

        theta_tilde = torch.distributions.MultivariateNormal(self.theta, self.precision_matrix).sample((batch_size,))  # type: ignore

        result = torch.argmax(
            torch.einsum("ijk,ik->ij", contextualised_actions, theta_tilde), dim=1
        )
        return torch.nn.functional.one_hot(result, num_classes=n_arms).reshape(
            -1, n_arms
        )


class LinearUCBBandit(LinearBandit):
    def __init__(self, n_features: int, alpha: float = 1.0) -> None:
        super().__init__(n_features)
        self.alpha = alpha

    def forward(self, contextualised_actions: torch.Tensor) -> torch.Tensor:
        assert (
            contextualised_actions.shape[2] == self.n_features
        ), "Contextualised actions must have shape (batch_size, n_arms, n_features)"
        n_arms = contextualised_actions.shape[1]

        print(
            self.alpha
            * torch.sqrt(
                torch.einsum(
                    "ijk,kl,ijl->ij",
                    contextualised_actions,
                    self.precision_matrix,
                    contextualised_actions,
                )
            )
        )

        result = torch.argmax(
            torch.einsum("ijk,k->ij", contextualised_actions, self.theta)
            + self.alpha
            * torch.sqrt(
                torch.einsum(
                    "ijk,kl,ijl->ij",
                    contextualised_actions,
                    self.precision_matrix,
                    contextualised_actions,
                )
            ),
            dim=1,
        )

        return torch.nn.functional.one_hot(result, num_classes=n_arms).reshape(
            -1, n_arms
        )


class LinearUCBApproxBandit(LinearUCBBandit):  # TODO
    pass


class LinearTSApproxBandit(LinearTSBandit):  # TODO
    pass
