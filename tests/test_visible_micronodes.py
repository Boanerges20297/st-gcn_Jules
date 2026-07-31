import unittest

import app as report_app
from shapely.geometry import shape


class VisibleMicronodesTest(unittest.TestCase):
    def test_default_response_keeps_all_eligible_micronodes(self):
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
            with report_app.app.test_request_context('/api/visible_micronodes?region=fortaleza'):
                response = report_app.get_visible_micronodes()
            features = response.get_json()['features']
            self.assertGreater(len(features), 40)
            self.assertEqual(1, calls)
            scores = [feature['properties']['score'] for feature in features]
            self.assertEqual(scores, sorted(scores, reverse=True))
            for feature in features:
                point = shape(feature['geometry']).centroid
                self.assertEqual('FORTALEZA', report_app._municipality_from_lnglat(point.x, point.y))
        finally:
            report_app._build_predictive_temporal_profiles = original_profiles
            report_app.rebuild_dynamic_micronode_exports = original_rebuild

    def test_explicit_limit_is_still_supported(self):
        original_rebuild = report_app.rebuild_dynamic_micronode_exports
        report_app.rebuild_dynamic_micronode_exports = lambda force=False: True
        try:
            with report_app.app.test_request_context('/api/visible_micronodes?region=fortaleza&limit=10'):
                response = report_app.get_visible_micronodes()
            self.assertEqual(10, len(response.get_json()['features']))
        finally:
            report_app.rebuild_dynamic_micronode_exports = original_rebuild


if __name__ == '__main__':
    unittest.main()
