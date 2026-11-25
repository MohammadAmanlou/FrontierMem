import numpy as np
from scipy import sparse
from scipy.optimize import check_grad
from scripts.train_lexical_baselines import objective
from frontiermem.dual_axis_protocol import ARMS
def test_lexical_objective_gradient():
    rng=np.random.default_rng(17)
    xr=sparse.csr_matrix(rng.normal(size=(6,5)))
    xs=sparse.csr_matrix(rng.normal(size=(4,5)))
    y=np.array([1,2,1,0,0,0])
    w=rng.normal(size=18)*.1
    for arm in ARMS:
        f=lambda v: objective(v,xr,y,xs,arm)[0]
        g=lambda v: objective(v,xr,y,xs,arm)[1]
        assert check_grad(f,g,w)<1e-5
