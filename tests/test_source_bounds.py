"""Regression checks for sources outside or on the receptor bounding box.

Run with pytest, or directly: python tests/test_source_bounds.py
"""
import unittest
import numpy as np
import pandas as pd
from FastGaussianPuff import GaussianPuff


class SourceBoundsTests(unittest.TestCase):
    def simulate(self, sensors, source, direction, using_sensors=True):
        options = dict(
            obs_dt=60, sim_dt=1, puff_dt=1, output_dt=60,
            simulation_start=pd.Timestamp("2026-02-01T12:00:00Z"),
            simulation_end=pd.Timestamp("2026-02-01T12:05:00Z"),
            time_zone="America/New_York", source_coordinates=np.array([source]),
            emission_rates=np.array([3.6]), wind_speeds=np.full(6, 2.25),
            wind_directions=np.full(6, direction),
            skip_low_wind=True, low_wind_cutoff=0.1, quiet=True,
        )
        if using_sensors:
            options.update(using_sensors=True, sensor_coordinates=sensors)
        else:
            options.update(using_sensors=False, grid_coordinates=np.array([10, 10, 0, 40, 40, 4]),
                           nx=4, ny=4, nz=3)
        return np.asarray(GaussianPuff(**options).simulate())

    def test_outside_sources_match_explicit_enclosing_receptors(self):
        # Relative coordinates from the Bryan and Old Crow problem geometries.
        geometries = [
            (np.array([[-9.479226, -73.231901, 2], [-80.759922, -43.194643, 2],
                       [44.511043, -3.700116, 2]]), [0, 0, 2]),
            (np.array([[139.030472, -37.244314, 2], [82.415772, 42.646162, 2],
                       [149.693588, 19.982196, 2]]), [0, 0, 2]),
        ]
        for sensors, source in geometries:
            # Explicit points create the same source-inclusive bounds as the fix.
            lo = np.minimum(sensors[:, :2].min(axis=0), np.array(source[:2]) - 1)
            hi = np.maximum(sensors[:, :2].max(axis=0), np.array(source[:2]) + 1)
            corners = np.array([[lo[0], lo[1], 2], [lo[0], hi[1], 2],
                                [hi[0], lo[1], 2], [hi[0], hi[1], 2]])
            for direction in [0, 90, 180, 270, 269.99817541606, 0.03062030813]:
                with self.subTest(source=source, sensors=sensors.tolist(), direction=direction):
                    actual = self.simulate(sensors, source, direction)
                    reference = self.simulate(np.vstack([sensors, corners]), source, direction)[:, :len(sensors)]
                    self.assertEqual(actual.shape, (6, len(sensors)))
                    self.assertTrue(np.isfinite(actual).all())
                    self.assertTrue((actual >= 0).all())
                    np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-10)

    def test_source_on_corner_or_degenerate_sensor_extent(self):
        for sensors in [np.array([[0, 0, 2], [20, 20, 2]]), np.array([[0, 0, 2]])]:
            for direction in [0, 90, 180, 270]:
                with self.subTest(sensors=sensors.tolist(), direction=direction):
                    result = self.simulate(sensors, [0, 0, 2], direction)
                    self.assertTrue(np.isfinite(result).all())
                    self.assertTrue((result >= 0).all())

    def test_grid_source_outside_domain(self):
        for direction in [0, 90, 180, 270]:
            with self.subTest(direction=direction):
                result = self.simulate(None, [0, 25, 2], direction, using_sensors=False)
                self.assertTrue(np.isfinite(result).all())
                self.assertTrue((result >= 0).all())
                if direction == 270:
                    self.assertGreater(result.max(), 0)


if __name__ == "__main__":
    unittest.main()
