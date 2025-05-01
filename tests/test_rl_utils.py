"""Test suite for reinforcement learning utility functions."""

import math
import unittest
import sys
import os

# Add the parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.rl_utils import (
    calculate_distance,
    discretize_distance,
    discretize_sensor,
    discretize_velocity,
    normalize_angle,
)


class TestRLUtils(unittest.TestCase):
    def test_calculate_distance(self):
        """Test Euclidean distance calculation."""
        p1 = [0.0, 0.0]
        p2 = [3.0, 4.0]
        self.assertEqual(calculate_distance(p1, p2), 5.0)

    def test_normalize_angle(self):
        """Test angle normalization to [-π, π] range."""
        self.assertAlmostEqual(normalize_angle(3 * math.pi), -math.pi)
        self.assertAlmostEqual(normalize_angle(-3 * math.pi), -math.pi)
        self.assertAlmostEqual(normalize_angle(0), 0)

    def test_discretize_distance(self):
        """Test distance discretization into bins."""
        self.assertEqual(discretize_distance(0.05), 0)  # First bin
        self.assertEqual(discretize_distance(0.15), 1)  # Second bin
        self.assertEqual(discretize_distance(3.0), 6)  # Beyond last bin

    def test_discretize_sensor(self):
        """Test sensor reading discretization."""
        self.assertEqual(discretize_sensor(50), 0)  # No obstacle
        self.assertEqual(discretize_sensor(200), 1)  # Far obstacle
        self.assertEqual(discretize_sensor(500), 2)  # Medium obstacle
        self.assertEqual(discretize_sensor(800), 3)  # Close obstacle

    def test_discretize_velocity(self):
        """Test velocity state discretization."""
        # Test stopped state
        self.assertEqual(discretize_velocity([0.0, 0.0]), 0)
        # Test forward motion
        self.assertEqual(discretize_velocity([3.0, 3.0]), 1)  # Slow forward
        self.assertEqual(discretize_velocity([8.0, 8.0]), 2)  # Fast forward
        # Test backward motion
        self.assertEqual(discretize_velocity([-2.0, -2.0]), 3)
        # Test turning
        self.assertEqual(discretize_velocity([2.0, -2.0]), 4)


if __name__ == "__main__":
    unittest.main()
