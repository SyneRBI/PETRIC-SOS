# PETRIC: PET Rapid Image reconstruction Challenge

## Stochastic Optimisation with Subsets (SOS)

Authors: Sam Porter 

Algorithm:
SAGA with a projected gradient descent warm start and armijo line searches.

SAGA minimises a sum of differentiable functions, $x^* = \underset{x}{\text{arg min}}\sum_n f_n(x)$. It approximates the gradient at each iteration by
$$
\begin{equation}
    g^{(k)} = N\left(\nabla_n f(x^{(k)}) - g_n (\phi^{(k)}_n)\right) + \sum_{n=0}^{N-1} g_n (\phi^{(k)}_n)
\end{equation}
$$
where N is the number of sub-functions.

A preconditioned Armijo line search is used to find a reasonable step size:
$$
\begin{align}
    &\text{prox}_{\eta^{(k)} h} \left (f_n(x^{(k)}) - \eta^{(k)} - \mathbf{P}^{(k)} \sum_n \nabla f_n(x^{(k)}) \right )  - \text{prox}_{\eta^{(k)} h} \left (\sum_n f_n(x^{(k)}) \right ) \nonumber \\  &\leq - \sigma \eta^{(k)} \left \langle \sum_n \nabla f_n(x^{(k)}) \;|\; \mathbf{P}^{(k)} \sum_n \nabla f_n(x^{(k)}) \right \rangle
\end{align}
$$
where $\sigma$ is a parameter enforcing sufficient decrease in the objective function along the preconditioned direction, $\mathbf{P}^{(k)}$, and $\eta^{(k)}$ is s step size that is decreased until the above Armijo conditions are fulfilles.

The algorithm is warm started with 5 iterations of preconditioned gradient descent usingt he full gradients and an Armijo line search in order to avoid over-estimation of step size. In order to improve convergence rate and guarantee convergence to a solution, the initil Armijo step size decays with $\dfrac{1}{1 + d * \text{iteration}}$. $d$ is a parameter that controls the rate of decay.

Acknowledgements: 
- Thanks to Margaret Duff, Casper da Costa-Luis and Edoardo Pasca for all their help
- This algorithm lies heavily on the stochastic implementations in CIL so thanks to all those wo've helped there, as well