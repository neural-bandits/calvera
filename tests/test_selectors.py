import pytest
import torch
from torch.testing import assert_close

from calvera.utils.selectors import (
    AbstractSelector,
    ArgMaxSelector,
    EpsilonGreedySelector,
    RandomSelector,
    TopKSelector,
    EpsilonGreedyTopKSelector,
)


class TestArgMaxSelector:
    def test_single_sample(self) -> None:
        selector = ArgMaxSelector()
        scores = torch.tensor([[1.0, 2.0, 3.0]])
        selected = selector(scores)

        expected = torch.tensor([[0, 0, 1]])
        assert_close(selected, expected)
        assert selected.shape == (1, 3)
        assert selected.sum() == 1  # Only one action selected

    def test_batch_samples(self) -> None:
        selector = ArgMaxSelector()
        scores = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 3.0, 1.0]])
        selected = selector(scores)

        expected = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        assert_close(selected, expected)
        assert selected.shape == (3, 3)
        assert selected.sum() == 3  # One action per sample

    def test_equal_scores(self) -> None:
        selector = ArgMaxSelector()
        scores = torch.tensor([[2.0, 2.0, 2.0]])
        selected = selector(scores)

        # ArgMax should select the first occurrence of maximum value
        expected = torch.tensor([[1, 0, 0]])
        assert_close(selected, expected)

    def test_negative_scores(self) -> None:
        selector = ArgMaxSelector()
        scores = torch.tensor([[-1.0, -2.0, -3.0]])
        selected = selector(scores)

        expected = torch.tensor([[1, 0, 0]])
        assert_close(selected, expected)

    def test_single_arm(self) -> None:
        selector = ArgMaxSelector()
        scores = torch.tensor([[1.0]])
        selected = selector(scores)

        expected = torch.tensor([[1]])
        assert_close(selected, expected)


class TestEpsilonGreedySelector:
    def test_initialization(self) -> None:
        # Test valid epsilon values
        EpsilonGreedySelector(0.0)
        EpsilonGreedySelector(0.5)
        EpsilonGreedySelector(1.0)

        # Test invalid epsilon values
        with pytest.raises(AssertionError):
            EpsilonGreedySelector(-0.1)
        with pytest.raises(AssertionError):
            EpsilonGreedySelector(1.1)

    def test_epsilon_zero(self) -> None:
        """When epsilon=0, should behave exactly like ArgMaxSelector"""
        selector = EpsilonGreedySelector(epsilon=0.0)
        scores = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        selected = selector(scores)

        expected = torch.tensor([[0, 0, 1], [1, 0, 0]])
        assert_close(selected, expected)
        assert selected.shape == (2, 3)
        assert selected.sum() == 2  # One action per sample

    def test_epsilon_one(self) -> None:
        """When epsilon=1, should always explore randomly"""
        selector = EpsilonGreedySelector(epsilon=1.0)
        scores = torch.tensor([[1.0, 2.0, 3.0]])

        selections = []
        for _ in range(100):
            selected = selector(scores)
            assert selected.shape == (1, 3)
            assert selected.sum() == 1  # One action selected
            selections.append(selected)

        stacked_selections = torch.stack(selections)
        assert (stacked_selections.sum(dim=0) > 0).all()  # All actions should be selected at least once

    @pytest.mark.parametrize("batch_size,n_arms", [(1, 3), (5, 2), (10, 4)])
    def test_output_shape_and_values(self, batch_size: int, n_arms: int) -> None:
        selector = EpsilonGreedySelector(epsilon=0.5)
        scores = torch.rand(batch_size, n_arms)
        selected = selector(scores)

        assert selected.shape == (batch_size, n_arms)
        assert selected.sum(dim=1).allclose(torch.ones(batch_size, dtype=torch.int64))  # One selection per sample
        assert ((selected == 0) | (selected == 1)).all()  # Only binary values


class TestTopKSelector:
    def test_initialization(self) -> None:
        # Test valid k values
        TopKSelector(k=1)
        TopKSelector(k=3)

        # Test invalid k values
        with pytest.raises(AssertionError):
            TopKSelector(k=0)
        with pytest.raises(AssertionError):
            TopKSelector(k=-1)

    def test_single_sample(self) -> None:
        selector = TopKSelector(k=2)
        scores = torch.tensor([[1.0, 4.0, 3.0, 2.0]])
        selected = selector(scores)

        expected = torch.tensor([[0, 1, 1, 0]])
        assert_close(selected, expected)
        assert selected.shape == (1, 4)
        assert selected.sum() == 2  # Exactly k actions selected

    def test_batch_samples(self) -> None:
        selector = TopKSelector(k=2)
        scores = torch.tensor([[1.0, 4.0, 3.0, 2.0], [4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 4.0, 3.0]])
        selected = selector(scores)

        expected = torch.tensor(
            [
                [0, 1, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
            ]
        )
        assert_close(selected, expected)
        assert selected.shape == (3, 4)
        assert (selected.sum(dim=1) == 2).all()  # Exactly k actions per sample

    def test_k_equals_n_arms(self) -> None:
        selector = TopKSelector(k=3)
        scores = torch.tensor([[1.0, 2.0, 3.0]])
        selected = selector(scores)

        expected = torch.tensor([[1, 1, 1]])
        assert_close(selected, expected)

    def test_k_too_large(self) -> None:
        selector = TopKSelector(k=4)
        scores = torch.tensor([[1.0, 2.0, 3.0]])  # Only 3 arms

        with pytest.raises(AssertionError):
            selector(scores)

    def test_equal_scores(self) -> None:
        selector = TopKSelector(k=2)
        scores = torch.tensor([[2.0, 2.0, 2.0, 2.0]])
        selected = selector(scores)

        # Should select first k occurrences of maximum value
        assert selected.sum() == 2
        assert selected[0, 0] == 1
        assert selected[0, 1] == 1
        assert selected[0, 2] == 0
        assert selected[0, 3] == 0

    @pytest.mark.parametrize("batch_size,n_arms,k", [(1, 5, 2), (3, 4, 1), (5, 6, 3), (10, 3, 2)])
    def test_output_shape_and_values(self, batch_size: int, n_arms: int, k: int) -> None:
        selector = TopKSelector(k=k)
        scores = torch.rand(batch_size, n_arms)
        selected = selector(scores)

        assert selected.shape == (batch_size, n_arms)
        assert (selected.sum(dim=1) == k).all()  # Exactly k selections per sample
        assert ((selected == 0) | (selected == 1)).all()  # Only binary values


class TestRandomSelector:
    # just test that it doesnt fail and the output shapes are correct:
    def test_random_selector(self) -> None:
        selector = RandomSelector()
        scores = torch.rand(10, 5)
        selected = selector(scores)

        assert selected.shape == (10, 5)
        assert (selected.sum(dim=1) == 1).all()
        assert ((selected == 0) | (selected == 1)).all()

    def test_random_selector_with_seed(self) -> None:
        selector = RandomSelector(seed=42)
        scores = torch.rand(10, 5)
        selected = selector(scores)

        assert selected.shape == (10, 5)
        assert (selected.sum(dim=1) == 1).all()
        assert ((selected == 0) | (selected == 1)).all()

        selector = RandomSelector(seed=42)
        selected2 = selector(scores)
        assert torch.equal(selected, selected2)


class TestSelectorSerialization:
    def test_argmax_selector_state_dict(self) -> None:
        """Test state dictionary creation for ArgMaxSelector."""
        selector = ArgMaxSelector()
        state = selector.get_state_dict()

        assert state["type"] == "ArgMaxSelector"
        # ArgMaxSelector has no additional state
        assert len(state) == 1

    def test_epsilon_greedy_selector_state_dict(self) -> None:
        """Test state dictionary creation for EpsilonGreedySelector."""
        epsilon = 0.15
        selector = EpsilonGreedySelector(epsilon=epsilon, seed=42)
        state = selector.get_state_dict()

        assert state["type"] == "EpsilonGreedySelector"
        assert state["epsilon"] == epsilon
        assert "generator_state" in state
        assert len(state) == 3

    def test_topk_selector_state_dict(self) -> None:
        """Test state dictionary creation for TopKSelector."""
        k = 3
        selector = TopKSelector(k=k)
        state = selector.get_state_dict()

        assert state["type"] == "TopKSelector"
        assert state["k"] == k
        assert len(state) == 2

    def test_from_state_dict_argmax(self) -> None:
        """Test creating ArgMaxSelector from state dictionary."""
        state = {"type": "ArgMaxSelector"}
        selector = AbstractSelector.from_state_dict(state)

        assert isinstance(selector, ArgMaxSelector)

    def test_from_state_dict_epsilon_greedy(self) -> None:
        """Test creating EpsilonGreedySelector from state dictionary."""
        epsilon = 0.2
        original = EpsilonGreedySelector(epsilon=epsilon, seed=42)
        original_state = original.generator.get_state()

        state = {"type": "EpsilonGreedySelector", "epsilon": epsilon, "generator_state": original_state}
        selector = AbstractSelector.from_state_dict(state)

        assert isinstance(selector, EpsilonGreedySelector)
        assert selector.epsilon == epsilon

        # Verify random generator was properly restored by comparing outputs
        scores = torch.tensor([[0.5, 0.8, 0.3]])

        original.generator.set_state(original_state)
        selector.generator.set_state(torch.as_tensor(state["generator_state"]))

        # Verify identical output with same random state
        assert torch.equal(original(scores), selector(scores))

    def test_from_state_dict_topk(self) -> None:
        """Test creating TopKSelector from state dictionary."""
        k = 2
        state = {"type": "TopKSelector", "k": k}
        selector = AbstractSelector.from_state_dict(state)

        assert isinstance(selector, TopKSelector)
        assert selector.k == k

    def test_from_state_dict_invalid_type(self) -> None:
        """Test error handling for unknown selector type."""
        state = {"type": "UnknownSelector"}
        with pytest.raises(ValueError, match="Unknown selector type"):
            AbstractSelector.from_state_dict(state)

    def test_round_trip_serialization(self) -> None:
        """Test round-trip serialization for all selector types."""
        orig_argmax = ArgMaxSelector()
        argmax_state = orig_argmax.get_state_dict()
        new_argmax = AbstractSelector.from_state_dict(argmax_state)
        assert isinstance(new_argmax, ArgMaxSelector)

        orig_eps = EpsilonGreedySelector(epsilon=0.3, seed=123)
        eps_state = orig_eps.get_state_dict()
        new_eps = AbstractSelector.from_state_dict(eps_state)
        assert isinstance(new_eps, EpsilonGreedySelector)
        assert new_eps.epsilon == orig_eps.epsilon

        orig_topk = TopKSelector(k=3)
        topk_state = orig_topk.get_state_dict()
        new_topk = AbstractSelector.from_state_dict(topk_state)
        assert isinstance(new_topk, TopKSelector)
        assert new_topk.k == orig_topk.k


class TestEpsilonGreedyTopKSelector:
    def test_initialization(self) -> None:
        """Test valid and invalid initialization parameters."""
        EpsilonGreedyTopKSelector(k=1, epsilon=0.0)
        EpsilonGreedyTopKSelector(k=3, epsilon=0.5)
        EpsilonGreedyTopKSelector(k=5, epsilon=1.0)

        # Test invalid k values
        with pytest.raises(AssertionError):
            EpsilonGreedyTopKSelector(k=0, epsilon=0.1)
        with pytest.raises(AssertionError):
            EpsilonGreedyTopKSelector(k=-1, epsilon=0.1)

        # Test invalid epsilon values
        with pytest.raises(AssertionError):
            EpsilonGreedyTopKSelector(k=1, epsilon=-0.1)
        with pytest.raises(AssertionError):
            EpsilonGreedyTopKSelector(k=1, epsilon=1.1)

    def test_epsilon_zero(self) -> None:
        """When epsilon=0, should behave exactly like TopKSelector"""
        scores = torch.tensor([[1.0, 4.0, 3.0, 2.0], [4.0, 3.0, 2.0, 1.0]])

        eg_selector = EpsilonGreedyTopKSelector(k=2, epsilon=0.0, seed=42)
        eg_selected = eg_selector(scores)

        topk_selector = TopKSelector(k=2)
        topk_selected = topk_selector(scores)

        # Results should be identical
        assert_close(eg_selected, topk_selected)

    def test_epsilon_one(self) -> None:
        """When epsilon=1, should always explore randomly"""
        selector = EpsilonGreedyTopKSelector(k=2, epsilon=1.0, seed=42)
        scores = torch.tensor([[1.0, 4.0, 3.0, 2.0]])  # Scores don't matter when epsilon=1

        selections = []
        for _ in range(100):
            selected = selector(scores)
            assert selected.shape == (1, 4)
            assert selected.sum() == 2
            selections.append(selected)

        # All arms should be selected at least once across the trials
        arm_selections = torch.sum(torch.stack(selections), dim=0)
        assert (arm_selections > 0).all()

    @pytest.mark.parametrize("batch_size,n_arms,k", [(1, 5, 2), (3, 4, 1), (5, 6, 3), (10, 3, 2)])
    def test_output_shape_and_values(self, batch_size: int, n_arms: int, k: int) -> None:
        """Test output shapes and values for various batch sizes, n_arms, and k values."""
        selector = EpsilonGreedyTopKSelector(k=k, epsilon=0.5, seed=42)
        scores = torch.rand(batch_size, n_arms)
        selected = selector(scores)

        assert selected.shape == (batch_size, n_arms)
        assert (selected.sum(dim=1) == k).all()  # Exactly k selections per sample
        assert ((selected == 0) | (selected == 1)).all()

    def test_k_too_large(self) -> None:
        """Test error when k is larger than n_arms."""
        selector = EpsilonGreedyTopKSelector(k=4, epsilon=0.5)
        scores = torch.tensor([[1.0, 2.0, 3.0]])

        with pytest.raises(AssertionError):
            selector(scores)

    def test_exploitation_exploration_balance(self) -> None:
        """Test that the selector balances between exploitation and exploration."""
        selector = EpsilonGreedyTopKSelector(k=1, epsilon=0.5, seed=123)

        scores = torch.tensor([[0.1, 0.9, 0.2, 0.3]])

        n_trials = 1000
        selections = torch.zeros(4)

        for _ in range(n_trials):
            selected = selector(scores)
            selections += selected[0]

        # The high-scoring arm (index 1) should be selected more often,
        # but all arms should have some selections due to exploration
        assert selections[1] > selections[0]  # Best arm selected more
        assert selections[1] > selections[2]
        assert selections[1] > selections[3]

        # All arms should have some selections
        assert (selections > 0).all()

        # Roughly 50% exploration, 50% exploitation
        # In exploitation, always arm 1
        # In exploration, each arm roughly 25% of the time
        # So arm 1 should be around 50% + (50% * 0.25) = ~62.5%
        assert 0.5 < selections[1] / n_trials < 0.75

    def test_serialization(self) -> None:
        """Test state dictionary creation and restoration."""
        k = 2
        epsilon = 0.15
        selector = EpsilonGreedyTopKSelector(k=k, epsilon=epsilon, seed=42)

        state = selector.get_state_dict()

        # Check state dict contents
        assert state["type"] == "EpsilonGreedyTopKSelector"
        assert state["k"] == k
        assert state["epsilon"] == epsilon
        assert "generator_state" in state

        new_selector = AbstractSelector.from_state_dict(state)

        assert isinstance(new_selector, EpsilonGreedyTopKSelector)
        assert new_selector.k == k
        assert new_selector.epsilon == epsilon

        # Verify identical behavior by running with same input
        scores = torch.tensor([[0.5, 0.8, 0.3, 0.7]])

        selector.generator.set_state(torch.as_tensor(state["generator_state"]))
        new_selector.generator.set_state(torch.as_tensor(state["generator_state"]))

        assert torch.equal(selector(scores.clone()), new_selector(scores.clone()))
