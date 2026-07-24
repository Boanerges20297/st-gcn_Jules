import unittest

from shapely.geometry import shape

import app


class GAHoneycombGridTest(unittest.TestCase):
    def test_ga_honeycomb_cells_do_not_overlap(self):
        radius = app._ga_honeycomb_radius_m()
        self.assertEqual(radius, 500)

        cells = []
        for q, r in ((0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)):
            x = radius * (3 ** 0.5) * (q + r / 2)
            y = radius * 1.5 * r
            cells.append(shape(app._hexagon_grid_geometry(x, y, radius, -3.8)))

        for index, left in enumerate(cells):
            for right in cells[index + 1:]:
                self.assertLess(left.intersection(right).area, 1e-14)
