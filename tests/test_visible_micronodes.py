import unittest

import app as report_app
from shapely.geometry import shape


class VisibleMicronodesTest(unittest.TestCase):
    def test_temporal_profile_is_built_once_per_response(self):
        calls = 0
        original_profiles = report_app._build_predictive_temporal_profiles
        original_rebuild = report_app.rebuild_dynamic_micronode_exports

        def counted_profiles():
            nonlocal calls
            calls += 1
            return original_profiles()

        report_app._build_predictive_temporal_profiles = counted_profiles
        report_app.rebuild_dynamic_micronode_exports = lambda force=False: True
        try:
            with report_app.app.test_request_context('/api/visible_micronodes?region=fortaleza&limit=2000'):
                response = report_app.get_visible_micronodes()
            features = response.get_json()['features']
            self.assertEqual(40, len(features))
            self.assertEqual(1, calls)
            scores = [feature['properties']['score'] for feature in features]
            self.assertEqual(scores, sorted(scores, reverse=True))
            for feature in features:
                point = shape(feature['geometry']).centroid
                self.assertEqual('FORTALEZA', report_app._municipality_from_lnglat(point.x, point.y))
        finally:
            report_app._build_predictive_temporal_profiles = original_profiles
            report_app.rebuild_dynamic_micronode_exports = original_rebuild


if __name__ == '__main__':
    unittest.main()
