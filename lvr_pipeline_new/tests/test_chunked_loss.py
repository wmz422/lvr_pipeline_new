import unittest

import torch
from torch import nn
from torch.nn import functional as F

from lvr.models.chunked_loss import ChunkedLMHead


def legacy_loss(head, hidden, targets):
    loss = torch.zeros((), device=hidden.device, dtype=torch.float32)
    for states, target in zip(hidden.split(256), targets.split(256)):
        loss = loss + F.cross_entropy(head(states).float(), target, reduction="sum")
    return loss / targets.numel()


class ChunkedLMHeadTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        context = torch.enable_grad()
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)

    def test_conversion_preserves_parameter_identity_state_and_rng(self):
        original = nn.Linear(8, 17).bfloat16().requires_grad_(False).eval()
        rng = torch.get_rng_state().clone()
        head = ChunkedLMHead.from_linear(original)
        self.assertIs(head.weight, original.weight)
        self.assertIs(head.bias, original.bias)
        self.assertFalse(head.training)
        self.assertFalse(head.weight.requires_grad)
        self.assertEqual(head.weight.dtype, torch.bfloat16)
        self.assertEqual(list(head.state_dict()), list(original.state_dict()))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertIs(ChunkedLMHead.from_linear(head), head)
        head.load_state_dict(original.state_dict(), strict=True)
        original.load_state_dict(head.state_dict(), strict=True)

    def test_tied_embedding_and_head_keep_the_same_parameter(self):
        embedding = nn.Embedding(17, 8)
        original = nn.Linear(8, 17, bias=False)
        original.weight = embedding.weight
        head = ChunkedLMHead.from_linear(original)
        self.assertIs(head.weight, embedding.weight)
        self.assertEqual(list(head.state_dict()), ["weight"])

    def test_logits_and_generation_shape_match_linear(self):
        for dtype in (torch.float32, torch.bfloat16):
            original = nn.Linear(8, 17).to(dtype)
            head = ChunkedLMHead.from_linear(original)
            hidden = torch.randn(2, 3, 8, dtype=dtype)
            torch.testing.assert_close(head(hidden), original(hidden), rtol=0, atol=0)
            self.assertEqual(head(hidden).shape, (2, 3, 17))

    def test_chunked_loss_and_all_gradients_match_original_loop(self):
        for dtype in (torch.float32, torch.bfloat16):
            for tokens in (1, 256, 257, 513, 7993, 8860):
                for frozen in (False, True):
                    with self.subTest(dtype=dtype, tokens=tokens, frozen=frozen):
                        original = nn.Linear(8, 17).to(dtype).requires_grad_(not frozen)
                        head = ChunkedLMHead.from_linear(original)
                        hidden = torch.randn(tokens, 8, dtype=dtype, requires_grad=True)
                        targets = torch.randint(17, (tokens,))
                        variables = (hidden,) if frozen else (hidden, head.weight, head.bias)
                        expected = legacy_loss(original, hidden, targets)
                        expected_grad = torch.autograd.grad(expected, variables)
                        actual = head(hidden, targets)
                        actual_grad = torch.autograd.grad(actual, variables)
                        self.assertEqual(actual.dtype, torch.float32)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        for result, reference in zip(actual_grad, expected_grad):
                            torch.testing.assert_close(result, reference, rtol=0, atol=0)

    def test_forward_backward_hooks_run_once_despite_different_chunk_counts(self):
        head = ChunkedLMHead(8, 17, bias=False)
        forward_calls, backward_calls = [], []
        head.register_forward_hook(lambda *args: forward_calls.append(1))
        head.register_full_backward_hook(lambda *args: backward_calls.append(1))
        for tokens in (7993, 8860):
            hidden = torch.randn(tokens, 8, requires_grad=True)
            head(hidden, torch.randint(17, (tokens,))).backward()
        self.assertEqual(len(forward_calls), 2)
        self.assertEqual(len(backward_calls), 2)
        # Frozen heads must still bracket the gradient path back to their inputs.
        head.requires_grad_(False)
        hidden = torch.randn(513, 8, requires_grad=True)
        head(hidden, torch.randint(17, (513,))).backward()
        self.assertGreater(hidden.grad.abs().sum(), 0)
        self.assertEqual(len(backward_calls), 3)

    def test_two_losses_in_one_backward_keep_gradients_and_hook_count(self):
        original = nn.Linear(8, 17, bias=False)
        head = ChunkedLMHead.from_linear(original)
        states = [torch.randn(count, 8, requires_grad=True) for count in (257, 7993)]
        targets = [torch.randint(17, (len(value),)) for value in states]
        variables = (*states, head.weight)
        expected = sum(legacy_loss(original, state, target) for state, target in zip(states, targets))
        expected_grad = torch.autograd.grad(expected, variables)
        calls = []
        head.register_full_backward_hook(lambda *args: calls.append(1))
        actual = sum(head(state, target) for state, target in zip(states, targets))
        actual_grad = torch.autograd.grad(actual, variables)
        self.assertEqual(len(calls), 2)
        for result, reference in zip(actual_grad, expected_grad):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)

    def test_invalid_loss_inputs_fail_clearly(self):
        head = ChunkedLMHead(8, 17, bias=False)
        with self.assertRaisesRegex(ValueError, "no supervised"):
            head(torch.empty(0, 8), torch.empty(0, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, "expects"):
            head(torch.empty(2, 8), torch.zeros(3, dtype=torch.long))
        with self.assertRaisesRegex(TypeError, "native nn.Linear"):
            ChunkedLMHead.from_linear(nn.Identity())


if __name__ == "__main__":
    unittest.main()
