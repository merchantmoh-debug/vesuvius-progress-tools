import unittest
import numpy as np
from depth_diagnostics import projection,depth_bounds,blended_profile,factor_swap,sample_offsets,sigmoid


class DepthDiagnosticsTests(unittest.TestCase):
    def test_logit_projection_is_not_probability_average(self):
        logits=np.array([-4.,2.]);weights=np.array([.25,.75])
        self.assertAlmostEqual(float(projection(logits,weights)),float(sigmoid(.5)))
        self.assertGreater(abs(float(projection(logits,weights))-float(weights@sigmoid(logits))),.03)

    def test_convex_bounds_and_blending(self):
        rng=np.random.default_rng(612)
        logits=rng.normal(size=(16,40));weights=rng.uniform(size=logits.shape);weights/=weights.sum(0)
        low,high=depth_bounds(logits);value=projection(logits,weights)
        self.assertTrue(np.all(value>=low) and np.all(value<=high))
        result=blended_profile([{'logits':[-5,-4],'attention':[.5,.5],'blend_weight':.4},
                                {'logits':[-3,-2],'attention':[.2,.8],'blend_weight':.6}])
        self.assertLessEqual(result['projected_probability'],result['upper_bound'])
        self.assertLess(result['upper_bound'],.1)

    def test_reject_invalid_projection_and_misaligned_physical_depths(self):
        with self.assertRaises(ValueError):projection([1,2],[-1,2])
        with self.assertRaises(ValueError):projection([1,2],[.2,.2])
        with self.assertRaises(ValueError):factor_swap([1,2],[.5,.5],[0,1],[1,2],[.5,.5],[1,2])

    def test_odd_even_and_reverse_depth_order(self):
        np.testing.assert_allclose(sample_offsets(28,13.5,9.362,[0,27]),[-126.387,126.387])
        np.testing.assert_allclose(sample_offsets(31,15,9.362,[7,23],True),[74.896,-74.896])

    def test_ct_shift_uses_physical_depth_and_reports_boundary(self):
        from ct_depth_alignment import bin_profile,fit_shift
        z=np.arange(-60,60.01,.25);values=np.sin(z*.09)+.3*np.cos(z*.24)
        centres=np.arange(-40,40.01,4);reference,_=bin_profile(z,values,centres+3,4)
        result=fit_shift(centres,reference,z,values,np.arange(-6,6.01,.5),4)
        self.assertEqual(result['shift_um'],3);self.assertFalse(result['at_search_boundary'])
        result=fit_shift(centres,reference,z,values,np.arange(-2,2.01,.5),4)
        self.assertTrue(result['at_search_boundary'])


if __name__=='__main__':unittest.main()
