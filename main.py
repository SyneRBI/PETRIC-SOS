"""Main file to modify for submissions.

Once renamed or symlinked as `main.py`, it will be used by `petric.py` as follows:

>>> from main import Submission, submission_callbacks
>>> from petric import data, metrics
>>> algorithm = Submission(data)
>>> algorithm.run(np.inf, callbacks=metrics + submission_callbacks)
"""
#%%
from cil.optimisation.algorithms import ISTA, Algorithm
from cil.optimisation.functions import IndicatorBox, SAGAFunction
from cil.optimisation.utilities import (Preconditioner, Sampler, 
                                        callbacks, StepSizeRule)
from petric import Dataset
from sirf.contrib.partitioner import partitioner
import numbers
import numpy as np
import pandas as pd
import types


assert issubclass(ISTA, Algorithm)

#%%
class MaxIteration(callbacks.Callback):
    """
    The organisers try to `Submission(data).run(inf)` i.e. for infinite iterations (until timeout).
    This callback forces stopping after `max_iteration` instead.
    """
    def __init__(self, max_iteration: int, verbose: int = 1):
        super().__init__(verbose)
        self.max_iteration = max_iteration

    def __call__(self, algorithm: Algorithm):
        if algorithm.iteration >= self.max_iteration:
            raise StopIteration

class MyPreconditioner(Preconditioner):
    """
    Example based on the row-sum of the Hessian of the log-likelihood. See: Tsai et al. Fast Quasi-Newton Algorithms
    for Penalized Reconstruction in Emission Tomography and Further Improvements via Preconditioning,
    IEEE TMI https://doi.org/10.1109/tmi.2017.2786865
    """
    def __init__(self, kappa):
        # add an epsilon to avoid division by zero (probably should make epsilon dependent on kappa)
        self.kappasq = kappa*kappa + ((kappa).max()/1000)**2

    def apply(self, algorithm, gradient, out=None):
        return gradient.divide(self.kappasq, out=out)
    
class LinearDecayStepSizeRule(StepSizeRule):
    """
    Linear decay of the step size.
    """
    def __init__(self, initial_step_size: float, decay: float):
        self.initial_step_size = initial_step_size
        self.decay = decay
        self.step_size = initial_step_size

    def get_step_size(self, algorithm):
        return self.initial_step_size / (1 + self.decay * algorithm.iteration)

#%%
class Submission(ISTA):
    """Stochastic variance reduced subset version of preconditioned ISTA"""

    # note that `issubclass(ISTA, Algorithm) == True`
    def __init__(self, data: Dataset):
        """
        Initialisation function, setting up data & (hyper)parameters.
        NB: in practice, `num_subsets` should likely be determined from the data.
        This is just an example. Try to modify and improve it!
        """

        num_subsets = data.acquired_data.dimensions()[2]//8
        initial_step_size = 0.5
        decay = (1/0.9-1) / num_subsets

        data_subs, acq_models, obj_funs = partitioner.data_partition(data.acquired_data, data.additive_term,
                                                                    data.mult_factors, num_subsets, mode='staggered',
                                                                    initial_image=data.OSEM_image)
        # WARNING: modifies prior strength with 1/num_subsets (as currently needed for BSREM implementations)
        data.prior.set_penalisation_factor(data.prior.get_penalisation_factor() / len(obj_funs))
        data.prior.set_up(data.OSEM_image)
        for f, d in zip(obj_funs, data_subs): # add prior to every objective function
            f.set_prior(data.prior)
            
        sampler = Sampler.random_without_replacement(len(obj_funs))
        preconditioner = MyPreconditioner(data.kappa)
        f = -SAGAFunction(obj_funs, sampler=sampler, snapshot_update_interval=None, store_gradients=True)
        g = IndicatorBox(lower=0, accelerated=True) # non-negativity constraint
        step_size_rule = LinearDecayStepSizeRule(initial_step_size, decay=decay)

        super().__init__(initial=data.OSEM_image, f=f, g=g, step_size=step_size_rule, preconditioner=preconditioner,
                         update_objective_interval=0)
        
submission_callbacks = [MaxIteration(np.inf)]