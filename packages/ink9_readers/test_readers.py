from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from oriented_context import orient_forward
from window_inference import infer_array,sliding_positions
from normalization_trace import point_trace


class MeanDepth(torch.nn.Module):
    def forward(self,x):return {'ink':x.mean(2)}


class ReaderTests(unittest.TestCase):
    def test_orientation_inverse_all_container_forms(self):
        value=torch.arange(48,dtype=torch.float32).reshape(1,1,2,4,6)
        for turns in range(4):
            for container in [None,list,tuple]:
                def forward(x):
                    out=x[:,:,0];return {'ink':out if container is None else container([out])}
                result=orient_forward(forward,turns,torch)(value)['ink']
                if container is not None:result=result[0]
                self.assertTrue(torch.equal(result,value[:,:,0]))

    def test_window_edges_constant_replay_and_depth_direction(self):
        self.assertEqual(sliding_positions(193,128,64),[0,64,65])
        raw=np.full((31,130,133),20,np.uint8)
        model=MeanDepth().eval()
        values,support,report=infer_array(model,lambda x:x.astype(np.float32)/255,raw,(17,128,128))
        np.testing.assert_allclose(values,1/(1+np.exp(-20/255)),atol=2e-7)
        self.assertEqual(report['tiles'],4);self.assertTrue(support.all())
        reverse,_,_=infer_array(model,lambda x:x.astype(np.float32)/255,raw,(17,128,128),reverse=True,quarter_turns=2)
        np.testing.assert_array_equal(values,reverse)

    def test_local_convolution_equal_but_instance_normalization_changes(self):
        torch.manual_seed(9)
        model=torch.nn.Sequential(torch.nn.Conv3d(1,2,3,padding=1,bias=False),torch.nn.InstanceNorm3d(2,affine=True)).eval()
        base=torch.randn(1,1,5,9,9);changed=base.clone();changed[:,:,:,:2]+=10
        _,a=point_trace(model,'0','1',base,(2,5,5))
        _,b=point_trace(model,'0','1',changed,(2,5,5))
        np.testing.assert_array_equal(a['convolution'],b['convolution'])
        self.assertGreater(np.max(abs(np.array(a['normalized'])-b['normalized'])),.01)
        self.assertLess(a['analytic_instance_norm_max_error'],1e-5)

if __name__=='__main__':unittest.main()
