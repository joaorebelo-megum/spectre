# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

from spectre.Domain.Creators import Brick


class TestBrick(unittest.TestCase):
    def test_construction(self):
        brick = Brick(
            lower_bounds=[1.0, 0.0, 2.0],
            upper_bounds=[2.0, 1.0, 3.0],
            is_periodic=[False, False, False],
            initial_refinement_levels=[1, 0, 1],
            initial_num_points=[3, 4, 2],
        )
        domain = brick.create_domain()
        self.assertFalse(domain.is_time_dependent())
        self.assertEqual(brick.block_names(), ["Brick"])
        self.assertEqual(brick.block_groups(), {})
        self.assertEqual(brick.initial_extents(), [[3, 4, 2]])
        self.assertEqual(brick.initial_refinement_levels(), [[1, 0, 1]])
        self.assertEqual(brick.functions_of_time(), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
