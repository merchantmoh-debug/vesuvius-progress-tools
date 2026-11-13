import gc
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.ndimage import gaussian_filter
from surface_geometry import bilinear,triangles,exact_surface_distances,grid_pitch,parent_motion,trim_invalid_border,world_to_grid
from fractional_sampling import affine_sample
from native_ct_planes import gather
from streamed_directions import build
from normal_alignment import controls


class SurfaceTests(unittest.TestCase):
    def plane(self):
        y,x=np.mgrid[:8,:9];return np.stack([10+x,20+y,np.full_like(x,30)],-1).astype(float)

    def test_exact_distance_and_native_quad_diagonal(self):
        g=self.plane();p=np.array([[12.2,23.4,35],[14,25,28]],float)
        np.testing.assert_allclose(exact_surface_distances(p,triangles(g)),[5,2],atol=1e-12)
        curved=np.array([[[0,0,0],[1,0,0]],[[0,1,0],[1,1,1]]],float)
        self.assertEqual(float(exact_surface_distances([[.5,.5,0]],triangles(curved))[0]),0)
        np.testing.assert_array_equal(bilinear(curved,[.5,.5]),[.5,.5,.25])

    def test_geometry_mapping_and_pitch(self):
        g=self.plane();point=bilinear(g,[2.25,3.5]);uv,error=world_to_grid(g[::-1,::-1],point,.001)
        np.testing.assert_allclose(uv,[5.75,3.5],atol=1e-8);self.assertLess(error,1e-8)
        pitch=grid_pitch(g,9.362)
        np.testing.assert_allclose(pitch['area_equivalent_pitch_um'],[9.362]*3)

    def test_immediate_parent_offset_and_padding_only_trim(self):
        g=self.plane();child=np.full((12,15,3),np.nan);child[3:11,4:13]=g
        np.testing.assert_array_equal(parent_motion(g,child,[4,3]),np.zeros(72))
        valid=np.zeros((12,15),bool);valid[3:11,4:13]=True
        cropped,offset=trim_invalid_border({'xyz':child,'generation':np.zeros(valid.shape)},valid)
        np.testing.assert_array_equal(cropped['xyz'],g);self.assertEqual(offset,[4,3])
        valid[5,6]=False
        with self.assertRaises(ValueError):trim_invalid_border({'xyz':child},valid)

    def test_fractional_affine_field_and_true_boundaries(self):
        z,y,x=np.mgrid[:18,:19,:20];volume=3*x+2*y+z
        a=affine_sample(volume,[1,2,3],np.eye(3),(3,4,5),trace_pitch_um=19.2,ct_pitch_um=8.64)
        zz,yy,xx=np.indices(a.shape);expected=3*(1+xx*19.2/8.64)+2*(2+yy*19.2/8.64)+(3+zz*19.2/8.64)
        np.testing.assert_allclose(a,expected,atol=1e-12)
        b=affine_sample(volume,[1,2,3],np.eye(3),(3,4,5),trace_pitch_um=1,ct_pitch_um=1)
        np.testing.assert_array_equal(b,volume[3:6,2:6,1:6])
        with self.assertRaises(ValueError):affine_sample(volume,[-1,0,0],np.eye(3),(2,2,2),trace_pitch_um=1,ct_pitch_um=1)

    def test_sparse_plane_gather_matches_full_cube_serial_and_parallel(self):
        volume=np.arange(12*12*12,dtype=np.uint16).reshape(12,12,12).astype(np.uint8)
        def acquire(key):
            z,y,x=np.array(key)*4;return volume[z:z+4,y:y+4,x:x+4]
        for workers in [1,4]:
            xy,yz,values,keys=gather([0,0,0],[12,12,12],[3,8],[5],[[2,3,4],[11,10,9]],acquire,4,workers)
            np.testing.assert_array_equal(xy[3],volume[3]);np.testing.assert_array_equal(xy[8],volume[8])
            np.testing.assert_array_equal(yz[5],volume[:,:,5]);np.testing.assert_array_equal(values,[volume[2,3,4],volume[11,10,9]])
            self.assertLess(len(keys),27)

    def test_streamed_gaussian_halos_match_complete_coefficients(self):
        volume=np.random.default_rng(4).normal(size=(65,19,17)).astype(np.float32)
        with tempfile.TemporaryDirectory() as folder:
            fields=build(volume,Path(folder),.8,2.,16)
            gy=gaussian_filter(volume,.8,order=(0,1,0),output=np.float32)
            for key,order in [('01',(0,0,1)),('11',None),('12',(1,0,0))]:
                other=gy if order is None else gaussian_filter(volume,.8,order=order,output=np.float32)
                expected=gaussian_filter(gy*other,2.,output=np.float32)
                np.testing.assert_array_equal(fields[key],expected)
            for value in fields.values():value._mmap.close()
            fields.clear();gc.collect()

    def test_normal_sign_conflicts_stay_explicit(self):
        self.assertTrue(all(value=='PASS' for value in controls().values()))


if __name__=='__main__':unittest.main()
