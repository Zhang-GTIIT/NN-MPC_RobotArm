"""Tests for ASAP timing helpers."""
from __future__ import annotations

import unittest

import numpy as np

from mpc.asap_timing import control_timing_sample


class ASAPTimingTests(unittest.TestCase):
    def test_period_wakeup_and_jitter_are_wall_clock_quantities(self) -> None:
        period, wakeup, jitter = control_timing_sample(10.012, 10.000, 10.010, 0.010)
        self.assertAlmostEqual(period, 0.012)
        self.assertAlmostEqual(wakeup, 0.002)
        self.assertAlmostEqual(jitter, 0.002)

    def test_first_tick_has_no_period_or_jitter(self) -> None:
        period, wakeup, jitter = control_timing_sample(1.0, None, 0.999, 0.01)
        self.assertTrue(np.isnan(period))
        self.assertTrue(np.isnan(jitter))
        self.assertAlmostEqual(wakeup, 0.001)


if __name__ == "__main__":
    unittest.main()
