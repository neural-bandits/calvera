import pytest
import torch
from torch.testing import assert_close

from neural_bandits.utils.selectors import AbstractSelector, ArgMaxSelector, EpsilonGreedySelector, TopKSelector


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
