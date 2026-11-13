import unittest
import json
from pathlib import Path
import numpy as np
import torch
from active_adamw import ActiveAdamW,parity_control
from cpu_attention import partitioned_attention
from cpu_winding_memory import tiled_detail_head
from fiber_guidance import retained_indices,compare_phase


class FittingTests(unittest.TestCase):
    def test_recorded_phase_comparison_uses_excluded_population(self):
        from pathlib import Path
        folder=Path(__file__).parent/'evidence'
        records=json.loads((folder/'fibre_spacing_result.json').read_text(encoding='utf-8'))['guides']
        with np.load(folder/'fibre_phases.npz',allow_pickle=False) as data:
            for record in records:
                name='guide'+record['guide']
                got=compare_phase(data[name+'_control'],data[name+'_dense'],data[name+'_spacing40'],data[name+'_spacing16'])
                self.assertEqual(got['evaluation_points'],record['arms']['control']['evaluation_points'])
                self.assertEqual(got['control_centre_phase'],record['arms']['control']['median_phase'])
                self.assertEqual(got['candidate_centre_phase'],record['arms']['dense']['median_phase'])
                self.assertLess(abs(1-got['p90_relative_reduction']-record['dense_to_control_p90_ratio']),1e-6)

    def test_original_optimizer_regression(self):
        self.assertEqual(parity_control()['status'],'PASS')

    def test_optimizer_tail_and_momentum_on_zero_gradient_steps(self):
        torch.manual_seed(3);old_threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            a=torch.nn.Parameter(torch.randn(49713));b=torch.nn.Parameter(a.detach().clone())
            dense=torch.optim.AdamW([a],lr=3e-5,weight_decay=0,foreach=False,fused=False)
            compact=ActiveAdamW(lr=3e-5,block_size=256)
            for i in range(9):
                g=torch.zeros_like(a)
                if i not in [0,4,7]:g[torch.tensor([0,255,256,4095,49712])]=torch.randn(5)
                a.grad=g.clone();dense.step();compact.step(b,g)
                self.assertEqual(a.detach().numpy().tobytes(),b.detach().numpy().tobytes())
                state=dense.state[a]
                self.assertEqual(compact.exp_avg.numpy().tobytes(),state['exp_avg'][compact.indices].numpy().tobytes())
                self.assertEqual(compact.exp_avg_sq.numpy().tobytes(),state['exp_avg_sq'][compact.indices].numpy().tobytes())
        finally:torch.set_num_threads(old_threads)

    def test_attention_preserves_heads_tokens_and_broadcast_masks(self):
        torch.manual_seed(4);q=torch.randn(7,3,19,8,dtype=torch.bfloat16)
        for mask in [None,torch.randn(19,19,dtype=q.dtype),torch.randn(3,19,19,dtype=q.dtype),torch.randn(7,3,19,19,dtype=q.dtype)]:
            gold=torch.nn.functional.scaled_dot_product_attention(q,q,q,attn_mask=mask)
            actual=partitioned_attention(q,q,q,attn_mask=mask,batch_limit=3)
            self.assertTrue(torch.equal(gold,actual))
        with self.assertRaises(AssertionError):partitioned_attention(q,q,q,dropout_p=.1)

    def test_detail_head_including_interpolation_seams(self):
        torch.manual_seed(7)
        class Model:
            full_resolution_head=torch.nn.Sequential(torch.nn.Conv3d(7,5,1),torch.nn.GELU(),torch.nn.Conv3d(5,2,1))
        model=Model();features=torch.randn(1,3,9,7,11);skip=torch.randn(1,4,18,14,11)
        with torch.inference_mode():
            expected=model.full_resolution_head(torch.cat([torch.nn.functional.interpolate(features,scale_factor=(2,2,1),mode='trilinear',align_corners=False),skip],1))
            actual=tiled_detail_head(model,features,skip,rows=8)
        torch.testing.assert_close(expected,actual,rtol=1e-6,atol=1e-7)

    def test_real_fibre_endpoints_and_excluded_evaluation_membership(self):
        root=Path(__file__).parent/'fiber_endpoint_fix'
        for name,count in [('guide_001.npy',79),('guide_002.npy',80)]:
            points=np.load(root/name);indices=retained_indices(points,16)
            self.assertEqual(len(indices),count);self.assertEqual(indices[-1],len(points)-1)
        report=compare_phase([0,9,2,8,4],[0,99,1,99,2],[1],[3],0)
        self.assertEqual(report['evaluation_points'],3)
        self.assertAlmostEqual(report['p90_relative_reduction'],.5)
        with self.assertRaises(ValueError):compare_phase([1],[1],[0],[],0)


if __name__=='__main__':unittest.main()
