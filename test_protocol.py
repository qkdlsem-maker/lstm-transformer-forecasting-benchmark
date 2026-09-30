import unittest
import numpy as np
import torch
from study import grid,Windows,make_model,count_params
from data_pipeline import prepare,finalize
from baselines import fit_ridge

class ProtocolTests(unittest.TestCase):
    def test_grid_and_parameter_counts(self):
        self.assertEqual(len(grid()),15)
        self.assertEqual(count_params(make_model('LSTM',24)),840872)
        self.assertEqual(count_params(make_model('Transformer',24)),836776)
    def test_forecast_window_has_no_future_input(self):
        a=np.arange(100,dtype=np.float32)[:,None].repeat(7,axis=1)
        x,y=Windows(a,[40],24,12)[0]
        self.assertEqual(x[-1,0].item(),39);self.assertEqual(y[0,0].item(),40)
    def test_ratio_evaluation_scale_and_origins_are_fixed(self):
        a,z,m,v,t=prepare('ETTh1',.3);b,zz,mm,vv,tt=prepare('ETTh1',1.)
        np.testing.assert_array_equal(v,vv);np.testing.assert_array_equal(t,tt)
        np.testing.assert_allclose(m['reference_scale'],mm['reference_scale'])
        self.assertGreater(m['train_start'],mm['train_start'])
        self.assertLessEqual(m['train_end'],v.min()-336)
    def test_identity_error_and_weighting(self):
        z=finalize(dict(abs=np.ones(7)*6,sq=np.ones(7)*12,count=3),np.ones(7)*2)
        self.assertAlmostEqual(z['mae'],1.);self.assertAlmostEqual(z['rmse'],1.)
    def test_ridge_fits_simple_linear_trajectory(self):
        a=np.arange(150,dtype=np.float32)[:,None].repeat(7,axis=1)/150
        fn=fit_ridge(a,np.arange(10,110),10,5,.01)
        p=fn(a[110:120][None]);np.testing.assert_allclose(p,a[120:125][None],atol=.005)
    def test_model_output_dimensions(self):
        for model in ['LSTM','Transformer']:
            m=make_model(model,12).eval()
            with torch.no_grad():self.assertEqual(m(torch.zeros(2,24,7)).shape,(2,12,7))

if __name__=='__main__':unittest.main()
