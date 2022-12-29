# PyTorch-QRNN
## Introduction
This repository contains a proof-of-concept implementation of Quasi Recurrent Neural Networks (QRNNs) [1].
The goal was to implement the recurrent pooling without a Python loop and without a custom CUDA kernel.
To this end, the recurrence is transformed to suitable `cumprod` and `cumsum` operations.
See below for the derivation.


## Derivation
The recurrent pooling stated in [1] is: 

$$h_t = f_t \cdot h_{t-1} + (1 - f_t) \cdot z_{t}$$

In order to calculate $h_t$ for all timesteps $t$ in a vectorized fashion, we expand the recursion as follows:

$$
\begin{align}
h_t 
&= f_t \cdot h_{t-1} + (1 - f_t) \cdot z_{t} \\
&= f_t \cdot f_{t-1} \cdot h_{t-2} + f_t \cdot (1 - f_{t-1}) \cdot z_{t-1} + (1 - f_t) \cdot z_{t} \\
&\ldots \\
&= \sum_{s=0}^t z_s \cdot (1 - f_s) \prod_{t'=s+1}^t f_{t'} \\
\end{align}
$$

Note that the initial hidden state is initialised to all zeros and therefore does not appear in the sum.
We cannot calculate the product of gates $\displaystyle\prod^t_{t'=s+1} f_{t'}$ efficiently, because this depends both on $t$ and $s$.
Therefore, computational and space complexity is quadratic in the number of timesteps.

To circumvent this problem, we first multiply the sum by $\displaystyle\prod_{t''=t+1}^T$, where $T$ is the number of timesteps, and divide afterwards.
By distributivity, this becomes:

$$
h_t = 
\sum_{s=0}^t z_s \cdot (1 - f_s) \prod_{t'=s+1}^t f_{t'} = 
\frac{\displaystyle\sum_{s=0}^{t} z_s \cdot (1 - f_s) \prod_{t'=s+1}^{T} f_{t'}}{\displaystyle \prod_{t'=t+1}^T f_{t'}}
$$

Now, the product $\displaystyle\prod_{t'=s+1}^t f_{t'}$, which is the same as in the denominator, can be pre-calculated efficiently independently of $t$ by a `cumprod` operation.
Accordingly, calculating the numerator is implemented by a `cumsum` operation.
Note, however, that the division prohibits any zeros in the recurrent gates. This is naturally enforced by using the `sigmoid` activation function to calculate recurrent gate values.
Unfortunately, this also prohibits any use of recurrent dropout such as zoneout explored in [1].

For numerical stability, however, we perform all computations in log space.
Accordingly, `cumprod` is replaced by `cumsum` and `cumsum` is replaced by `logcumsumexp`.
Note that this way, the activation function used to calculate $z$ has to be strictly positive, such as `exp` or `softplus`.

## Experiments
TODO

## References

[1]: Bradbury, James, et al. "Quasi-recurrent neural networks." arXiv preprint arXiv:1611.01576 (2016). 
