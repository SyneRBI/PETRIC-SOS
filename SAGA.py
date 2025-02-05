"""Main file to modify for submissions.

Once renamed or symlinked as `main.py`, it will be used by `petric.py` as follows:

>>> from main import Submission, submission_callbacks
>>> from petric import data, metrics
>>> algorithm = Submission(data)
>>> algorithm.run(np.inf, callbacks=metrics + submission_callbacks)
"""
from cil.optimisation.algorithms import ISTA, Algorithm
from cil.optimisation.functions import IndicatorBox
from cil.optimisation.functions import SGFunction, SAGFunction, SAGAFunction,\
                                        SVRGFunction, LSVRGFunction
from cil.optimisation.utilities import ConstantStepSize, Preconditioner, Sampler, callbacks, StepSizeRule
from petric import Dataset
from sirf.contrib.partitioner import partitioner

import numpy as np
import os

assert issubclass(ISTA, Algorithm)


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
        self.kappasq = kappa*kappa + 1e-6

    def apply(self, algorithm, gradient, out=None):
        return gradient.divide(self.kappasq, out=out)


class DecreasingStepSize(StepSizeRule):
    
    def __init__(self, initial_step_size, num_subsets, relaxation_par=0.02):
        self.iteration_num = 0
        self.num_subsets = num_subsets
        self.initial_step_size = initial_step_size
        self.relaxation_par = relaxation_par

    def get_step_size(self, algorithm):
        self.iteration_num += 1
        # THINK: we should not depend on the number of subsets so strongly
        # the relaxation parameter could be independent of the number of subsets
        # tmp_epoch = self.iteration_num // (self.num_subsets/2)
        # step_size = self.initial_step_size / (1. + 0.1 * tmp_epoch) 
        
        step_size = self.initial_step_size / (1.0 + self.relaxation_par * self.iteration_num)
        
        return step_size

# TODO Add callback to save function.data_passes as npy array


class Submission(ISTA):
    """Stochastic subset version of preconditioned ISTA"""

    # note that `issubclass(ISTA, Algorithm) == True`
    def __init__(self, data: Dataset, num_subsets: int = 21, step_size: float = 0.1,
                 update_objective_interval: int = 10, 
                 stochastic_function="SAGAFunction",
                 function_kwargs={}):
        """
        Initialisation function, setting up data & (hyper)parameters.
        NB: in practice, `num_subsets` should likely be determined from the data.
        This is just an example. Try to modify and improve it!
        """
        import logging
        logger = logging.getLogger(__name__)
        
        data_sub, acq_models, obj_funs = partitioner.data_partition(data.acquired_data, data.additive_term,
                                                                    data.mult_factors, num_subsets, mode='staggered',
                                                                    initial_image=data.OSEM_image)
        # WARNING: modifies prior strength with 1/num_subsets (as currently needed for ISTA implementations)
        data.prior.set_penalisation_factor(data.prior.get_penalisation_factor() / len(obj_funs))
        data.prior.set_up(data.OSEM_image)
        for f in obj_funs: # add prior evenly to every objective function
            f.set_prior(data.prior)

        sampler = Sampler.random_with_replacement(len(obj_funs), seed=42)

        # f = -SGFunction(obj_funs, sampler=sampler)   # negative to turn minimiser into maximiser
        # f = -SAGAFunction(obj_funs, sampler=sampler)
        logger.info(f"Using {stochastic_function} with args {function_kwargs}")
        logger.info(f"Using {num_subsets} subsets and initial step size {step_size}")
        f = -eval(stochastic_function)(obj_funs, sampler=sampler, **function_kwargs)
        if stochastic_function in ['SAGFunction', 'SAGAFunction']:
            f.function.warm_start_approximate_gradients(data.OSEM_image)

        step_size_rule = DecreasingStepSize(initial_step_size=step_size, num_subsets=num_subsets)
        g = IndicatorBox(lower=0, accelerated=False) # non-negativity constraint

        preconditioner = MyPreconditioner(data.kappa)
        super().__init__(initial=data.OSEM_image, f=f, g=g, step_size=step_size_rule, preconditioner=preconditioner,
                         update_objective_interval=update_objective_interval)


# submission_callbacks = [ MaxIteration(NUM_SUBSETS) ]
submission_callbacks = []

