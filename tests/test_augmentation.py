import unittest

import torch
from einops import repeat

from src.data_augmentation import FlipAndRotateSpace


class TestAugmentation(unittest.TestCase):
    def test_flip_and_rotate_space(self):
        aug = FlipAndRotateSpace(enabled=True)
        space_x = torch.randn(100, 10, 10, 3)  # (b, h, w, c)
        space_time_x = repeat(space_x.clone(), "b h w c -> b h w t c", t=8)
        new_space_time_x, new_space_x = aug.apply(space_time_x, space_x)

        # check that space_x and space_time_x are transformed the *same* way
        self.assertTrue(torch.equal(new_space_time_x.mean(dim=-2), new_space_x))

        # check that tensors were changed when flip+rotate=True
        self.assertFalse(torch.equal(new_space_time_x, space_time_x))
        self.assertFalse(torch.equal(new_space_x, space_x))

        aug = FlipAndRotateSpace(enabled=False)
        space_x = torch.randn(100, 10, 10, 3)  # (b, h, w, c)
        space_time_x = repeat(space_x.clone(), "b h w c -> b h w t c", t=8)
        new_space_time_x, new_space_x = aug.apply(space_time_x, space_x)

        # check that tensors were not changed when flip+rotate=False
        self.assertTrue(torch.equal(new_space_time_x, space_time_x))
        self.assertTrue(torch.equal(new_space_x, space_x))
