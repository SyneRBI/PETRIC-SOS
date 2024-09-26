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
                                        StepSizeRule)
from petric import Dataset
from sirf.contrib.partitioner import partitioner
import sirf.STIR as pet
import numpy as np

assert issubclass(ISTA, Algorithm)

class BSREMPreconditioner(Preconditioner):
    '''Step size rule for BSREM algorithm.
    
    ::math::
        x^+ = x + t \nabla \log L(y|x)
        
    with :math:`t = x / s` where :math:`s` is the adjoint of the range geometry of the acquisition model.
    '''
    def __init__(self, acq_models, freeze_iter = np.inf, epsilon=1e-6):

        self.epsilon = epsilon
        self.freeze_iter = freeze_iter
        self.t = None

        for i,el in enumerate(acq_models):
            if i == 0:
                self.s_sum = el.domain_geometry().get_uniform_copy(0.)
            ones = el.range_geometry().allocate(1.)
            s = el.adjoint(ones)
            s.maximum(self.epsilon, out=s)
            arr = s.as_array()
            np.reciprocal(arr, out=arr)
            s.fill(arr)
            self.s_sum += s
    
    def apply(self, algorithm, gradient, out=None):
        
        if algorithm.iteration < self.freeze_iter:
            t = algorithm.solution * self.s_sum + self.epsilon
        else:
            if self.t is None:
                self.t = algorithm.solution * self.s_sum + self.epsilon
            t = self.t

        return gradient.multiply(t, out=out)
    
    def apply_without_algorithm(self, gradient, x, out=None):
        t = x * self.s_sum + self.epsilon
        return gradient.multiply(t, out=out)

class LinearDecayStepSizeRule(StepSizeRule):
    """
    Linear decay of the step size with iteration.
    """
    def __init__(self, initial_step_size: float, decay: float):
        self.initial_step_size = initial_step_size
        self.decay = decay
        self.step_size = initial_step_size

    def get_step_size(self, algorithm):
        return self.initial_step_size / (1 + self.decay * algorithm.iteration)
    
def armijo_step_size_search_rule(x, f, g, grad, precond_grad, step_size=2.0, beta = 0.5, max_iter=100, tol=0.2):
    """
    Simple line search for the initial step size.
    """
    f_x = f(x) + g(x)
    g_norm = grad.dot(precond_grad)
    for _ in range(max_iter):
        x_new = g.proximal(x - step_size * precond_grad, step_size)
        f_x_new = f(x_new) + g(x_new)
        if f_x_new <= f_x - tol * step_size * g_norm:
            break
        step_size *= beta
    return step_size

def calculate_subsets(sino, min_counts_per_subset=2**20, max_subsets=30):
    """
    Calculate the number of subsets for a given sinogram such that each subset
    has at least the minimum number of counts.

    Args:
        sino: A sinogram object with .dimensions() and .sum() methods.
        min_counts_per_subset (float): Minimum number of counts per subset (default is 11057672.26).

    Returns:
        int: The number of subsets that can be created while maintaining the minimum counts per subset.
    """
    views = sino.dimensions()[2]  # Extract the number of views
    total_counts = sino.sum()     # Sum of counts for the sinogram
    
    # Calculate the maximum number of subsets based on minimum counts per subset
    max_subsets = int(total_counts / min_counts_per_subset)
    # ensure less than views / 4 subsets
    max_subsets = min(max_subsets, views // 4)
    # ensure less than max_subsets
    max_subsets = min(max_subsets, max_subsets)

    # Find a divisor of the number of views that results in the closest number of subsets
    subsets = max(1, min(views, max_subsets))

    # Ensure subsets is a divisor of views
    while views % subsets != 0 and subsets > 1:
        subsets -= 1
    
    return subsets

class Submission(ISTA):
    """Stochastic variance reduced subset version of preconditioned ISTA"""

    # note that `issubclass(ISTA, Algorithm) == True`
    def __init__(self, data: Dataset):
        """
        Initialisation function, setting up data & (hyper)parameters.
        """
        
        # Very simple heuristic to determine the number of subsets
        self.num_subsets = calculate_subsets(data.acquired_data, min_counts_per_subset=2**20)   
        update_interval = self.num_subsets
        # 10% decay per update interval
        upper_decay_perc = 0.1
        upper_decay = (1/(1-upper_decay_perc) - 1)/update_interval
        beta = 0.5

        data_subs, acq_models, obj_funs = partitioner.data_partition(data.acquired_data, data.additive_term,
                                                                    data.mult_factors, self.num_subsets, mode='staggered',
                                                                    initial_image=data.OSEM_image)

        
        data.prior.set_penalisation_factor(data.prior.get_penalisation_factor() / len(obj_funs))
        data.prior.set_up(data.OSEM_image)

        grad = data.OSEM_image.get_uniform_copy(0)
        
        for f, d in zip(obj_funs, data_subs): # add prior to every objective function
            f.set_prior(data.prior)
            grad -= f.gradient(data.OSEM_image)
            
        sampler = Sampler.random_without_replacement(len(obj_funs))
        f = -SAGAFunction(obj_funs, sampler=sampler, snapshot_update_interval=update_interval, store_gradients=True)

        preconditioner = BSREMPreconditioner(acq_models, epsilon=data.OSEM_image.max()/1e6, freeze_iter=10*update_interval)
        g = IndicatorBox(lower=0, accelerated=True) # non-negativity constraint
        
        precond_grad = preconditioner.apply_without_algorithm(grad, data.OSEM_image)
            
        initial_step_size = armijo_step_size_search_rule(data.OSEM_image, f, g, grad, precond_grad, beta=beta, step_size = 0.08, tol=0.2)
        step_size_rule = LinearDecayStepSizeRule(initial_step_size, 0.01)
        
        super().__init__(initial=data.OSEM_image, f=f, g=g, step_size=step_size_rule, 
                         preconditioner=preconditioner, update_objective_interval=update_interval)
        
submission_callbacks = []