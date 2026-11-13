import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from groupnorm_probe import compare,FrozenGroupNorm
from readonly_weights import write_state,ReadOnlyState


class CanonicalToolsTests(unittest.TestCase):
    def test_normalization_intervention_and_hook_cleanup(self):
        torch.manual_seed(2)
        model=torch.nn.Sequential(torch.nn.Conv3d(1,4,1),torch.nn.GroupNorm(2,4)).eval()
        base=torch.randn(1,1,4,6,6);candidate=base.clone();candidate[:,:,:,:2]+=5
        mask=torch.zeros(1,4,4,6,6,dtype=torch.bool);mask[:,:,:,3:]=True
        outputs,report=compare(model,base,candidate,mask=mask,replay_tolerance=1e-6)
        self.assertGreater(report['ordinary_mean_change'],.01)
        self.assertLess(report['frozen_mean_change'],1e-6)
        self.assertEqual(len(model[1]._forward_hooks),0)
        self.assertEqual(set(outputs),{'baseline_original','baseline_frozen','candidate_original','candidate_frozen'})

    def test_exception_removes_hooks(self):
        model=torch.nn.GroupNorm(2,4).eval()
        with self.assertRaises(RuntimeError):
            with FrozenGroupNorm(model):raise RuntimeError('test')
        self.assertEqual(len(model._forward_hooks),0)

    def test_readonly_weights_roundtrip_and_inference(self):
        torch.manual_seed(6)
        original=torch.nn.Linear(7,3).eval();expected=original(torch.ones(2,7)).detach()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'weights.safetensors';report=write_state(original.state_dict(),path)
            state=ReadOnlyState(path)
            with torch.device('meta'):actual=torch.nn.Linear(7,3)
            state.load_into(actual)
            with torch.inference_mode():self.assertTrue(torch.equal(actual(torch.ones(2,7)),expected))
            self.assertEqual(report['tensors'],2)
            self.assertTrue(all(not p.requires_grad for p in actual.parameters()))
            del actual
            state.tensors.clear();state.mapping.close();state.handle.close()

    def test_zero_bias_repair_preserves_real_bias_and_rejects_nonzero(self):
        import onnx
        from onnx import helper,numpy_helper,TensorProto
        from repair_zero_bias import repair
        def graph(bad):
            tensors=[numpy_helper.from_array(np.ones((1,1,1,1),np.float32),'a.weight'),
                     numpy_helper.from_array(np.array([[bad,0]],np.float32),'bad'),
                     numpy_helper.from_array(np.array([.2],np.float32),'real')]
            return helper.make_model(helper.make_graph([
                helper.make_node('Conv',['input','a.weight','bad'],['mid'],name='bad_conv'),
                helper.make_node('Conv',['mid','a.weight','real'],['output'],name='real_conv')],
                'test',[helper.make_tensor_value_info('input',TensorProto.FLOAT,[1,1,2,2])],
                [helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,1,2,2])],tensors),
                opset_imports=[helper.make_opsetid('',18)])
        m=graph(0);before=[x.SerializeToString() for x in m.graph.initializer]
        result=repair(m,{'a.weight'},Path('.'));onnx.checker.check_model(m)
        self.assertEqual(len(result),1);self.assertEqual(list(m.graph.node[1].input),['mid','a.weight','real'])
        self.assertEqual(before,[x.SerializeToString() for x in m.graph.initializer])
        unnamed=graph(0)
        for node in unnamed.graph.node:node.name=''
        repair(unnamed,{'a.weight'},Path('.'))
        self.assertEqual(list(unnamed.graph.node[1].input),['mid','a.weight','real'])
        with self.assertRaises(ValueError):repair(graph(1),{'a.weight'},Path('.'))
        with self.assertRaises(ValueError):repair(graph(0),{'a.weight','a.bias'},Path('.'))

    def test_directml_preparation_and_postprocessing_contract(self):
        from directml_runtime import predict
        class FixedGraph:
            def run(self,names,feeds):
                self.feed=feeds['volume'];return [np.zeros((1,1,64,64),np.float32)]
        session=FixedGraph();raw=np.full((62,256,256),255,np.uint8)
        result=predict(session,raw)
        np.testing.assert_array_equal(session.feed,np.ones((1,1,62,256,256),np.float32))
        np.testing.assert_array_equal(result,np.full((256,256),.5,np.float32))


if __name__=='__main__':unittest.main()
