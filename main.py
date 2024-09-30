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
from sirf.contrib.partitioner import partitioner
from petric import Dataset
import numpy as np

assert issubclass(ISTA, Algorithm)

class FullGradientInitialiserFunction(SAGAFunction):
    
    def __init__(self, functions, sampler=None, init_steps=0, **kwargs):
        
        super(FullGradientInitialiserFunction, self).__init__(functions, sampler=sampler, **kwargs)
        self.counter = 0
        self.init_steps = init_steps
        
    def gradient(self, x, out=None):
        """ Selects a random function using the `sampler` anad then calls the approximate gradient at :code:`x`

        Parameters
        ----------
        x : DataContainer
        out: return DataContainer, if `None` a new DataContainer is returned, default `None`.

        Returns
        --------
        DataContainer
            the value of the approximate gradient of the sum function at :code:`x`   
        """
        
        while self.counter < self.init_steps:
            self.counter += 1
            return self.full_gradient(x, out=out)

        self.function_num = self.sampler.next()
        
        self._update_data_passes_indices([self.function_num])
        
        return self.approximate_gradient(x, self.function_num, out=out)

class BSREMPreconditioner(Preconditioner):
    '''
    Preconditioner for BSREM
    '''
    
    def __init__(self, obj_funs, freeze_iter = np.inf, epsilon=1e-6):

        self.epsilon = epsilon
        self.freeze_iter = freeze_iter
        self.freeze = None

        for i,el in enumerate(obj_funs):
            s_inv = el.get_subset_sensitivity(0)
            s_inv.maximum(0, out=s_inv)
            arr = s_inv.as_array()
            np.reciprocal(arr, out=arr, where=arr!=0)
            s_inv.fill(arr)
            if i == 0:
                self.s_sum_inv = s_inv
            else:
                self.s_sum_inv += s_inv
        
    def apply(self, algorithm, gradient, out=None):
        if algorithm.iteration < self.freeze_iter:
            ret = gradient * ((algorithm.solution * self.s_sum_inv) + self.epsilon)
        else:
            if self.freeze is None:
                self.freeze = ((algorithm.solution * self.s_sum_inv) + self.epsilon)
            ret =  gradient * self.freeze
        if out is not None:
            out.fill(ret)
        else:
            return ret

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

class ArmijoStepSearchRule(StepSizeRule):
    """
    Armijo rule for step size for initial steps, followed by linear decay.
    """
    def __init__(self, initial_step_size: float, beta: float, decay: float, max_iter: int, tol: float, init_steps: int, update_interval=1):
        
        self.initial_step_size = initial_step_size
        self.step_size = initial_step_size
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.steps = init_steps
        self.update_interval = update_interval
        self.counter = 0
        self.linear_decay = None
        self.decay = decay
        self.f_x = None

    def get_step_size(self, algorithm):
        """
        Calculate and return the step size based on the Armijo rule.
        Step size is updated every `update_interval` iterations or during the initial steps.

        After Armijo iterations are exhausted, linear decay is applied.
        """
        # Check if we're within the initial steps or at an update interval
        if self.counter < self.steps: # or algorithm.iteration == self.update_interval:
            if self.f_x is None:
                self.f_x = algorithm.f(algorithm.solution) + algorithm.g(algorithm.solution)
            precond_grad = algorithm.preconditioner.apply(algorithm, algorithm.gradient_update)
            g_norm = algorithm.gradient_update.dot(precond_grad)
            
            # Reset step size to initial value for the Armijo search
            step_size = self.initial_step_size
            
            # Armijo step size search
            for _ in range(self.max_iter):
                # Proximal step
                x_new = algorithm.solution.copy().sapyb(1, precond_grad, -step_size)
                algorithm.g.proximal(x_new, step_size, out=x_new)
                f_x_new = algorithm.f(x_new) + algorithm.g(x_new)
                # Armijo condition check
                if f_x_new <= self.f_x - self.tol * step_size * g_norm:
                    self.f_x = f_x_new
                    break
                
                # Reduce step size
                step_size *= self.beta
            
            # Update the internal state with the new step size as the minimum of the current and previous step sizes
            self.step_size = min(step_size, self.step_size)
            
            self.initial_step_size = self.step_size
            
            if self.counter < self.steps:
                self.counter += 1
            
            return step_size

        # Apply linear decay if Armijo steps are done
        if self.linear_decay is None: # or algorithm.iteration == self.update_interval:
            self.f_x = None
            self.linear_decay = LinearDecayStepSizeRule(self.step_size, self.decay)
        
        # Return decayed step size
        return self.linear_decay.get_step_size(algorithm)
    
def calculate_subsets(sino, min_counts_per_subset=2**20, max_num_subsets=16):
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
    max_subsets = min(max_subsets, max_num_subsets)
    # Find a divisor of the number of views that results in the closest number of subsets
    subsets = max(1, min(views, max_subsets))

    # Ensure subsets is a divisor of views
    while views % subsets != 0 and subsets > 1:
        subsets -= 1
    
    return subsets

def update(self):
    r"""Performs a single iteration of ISTA with the preconditioner step separated

    .. math:: x_{k+1} = \mathrm{prox}_{\alpha g}(x_{k} - \alpha\nabla f(x_{k}))

    """
    self.f.gradient(self.x_old, out=self.gradient_update)
    
    try:
        step_size = self.step_size_rule.get_step_size(self)
    except NameError:
        raise NameError(msg='`step_size` must be `None`, a real float or a child class of :meth:`cil.optimisation.utilities.StepSizeRule`')

    self.x_old.sapyb(1., self.preconditioner.apply(self, self.gradient_update), -step_size, out=self.x_old)

    # proximal step
    self.g.proximal(self.x_old, step_size, out=self.x)
    
ISTA.update = update

class Submission(ISTA):
    """Stochastic variance reduced subset version of preconditioned ISTA"""

    # note that `issubclass(ISTA, Algorithm) == True`
    def __init__(self, data: Dataset, update_objective_interval=10):
        """
        Initialisation function, setting up data & (hyper)parameters.
        """
        # Very simple heuristic to determine the number of subsets
        self.num_subsets = calculate_subsets(data.acquired_data, min_counts_per_subset=2**20, max_num_subsets=16) 
        update_interval = self.num_subsets
        # 10% decay per update interval
        decay_perc = 0.1
        decay = (1/(1-decay_perc) - 1)/update_interval
        beta = 0.5
        
        print(f"Using {self.num_subsets} subsets")

        _, _, obj_funs = partitioner.data_partition(data.acquired_data, data.additive_term,
                                                                    data.mult_factors, self.num_subsets, mode='staggered',
                                                                    initial_image=data.OSEM_image)
        print("made it past partitioner")
        
        data.prior.set_penalisation_factor(data.prior.get_penalisation_factor() / len(obj_funs))
        data.prior.set_up(data.OSEM_image)
        
        for f in obj_funs: # add prior evenly to every objective function
            f.set_prior(data.prior)
            
        sampler = Sampler.random_without_replacement(len(obj_funs))
        f = -FullGradientInitialiserFunction(obj_funs, sampler=sampler, init_steps=5)

        preconditioner = BSREMPreconditioner(obj_funs, epsilon=data.OSEM_image.max()/1e6, freeze_iter=10*update_interval+5)
        g = IndicatorBox(lower=0, accelerated=True) # non-negativity constraint
            
        step_size_rule = ArmijoStepSearchRule(0.08, beta, decay, max_iter=100, tol=0.2, init_steps=5, update_interval=10*update_interval+5)
        
        super().__init__(initial=data.OSEM_image, f=f, g=g, step_size=step_size_rule, 
                         preconditioner=preconditioner, update_objective_interval=update_objective_interval)
        
submission_callbacks = []