# PyTorch-QRNN
## Introduction
This repository contains a proof-of-concept implementation of Quasi Recurrent Neural Networks (QRNNs) [1].
The goal was to implement the recurrent pooling without a Python loop and without a custom CUDA kernel.
To this end, the recurrence is transformed to suitable `cumprod` and `cumsum` operations.
See below for the derivation.

Please note that this implementation is still about 2x slower than the PyTorch LSTM implementation (on a single GPU).
Also, this implementation does not enable bidirectional sequence processing.
However, this can be achieved by appropriate masking and flipping of gates.


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

Now, the product $\displaystyle\prod_{t'=s+1}^T f_{t'}$, which is the same as in the denominator, can be pre-calculated efficiently independently of $t$ by a `cumprod` operation.
Accordingly, calculating the numerator is implemented by a `cumsum` operation.
Note, however, that the division prohibits any zeros in the recurrent gates. This is naturally enforced by using the `sigmoid` activation function to calculate recurrent gate values.

For numerical stability, however, we perform all computations in log space.
Accordingly, `cumprod` is replaced by `cumsum` and `cumsum` is replaced by `logcumsumexp`.
Note that this way, the activation function used to calculate $z$ has to be strictly positive, such as `exp` or `softplus`.

## Usage
The file `qrnn_layer.py` contains a class implementing a QRNN layer as described in [1].
It takes the following arguments:
  * `input_size`: Number of input features / channels
  * `hidden_size`: Number of output features / channels
  * `kernel_size`: Convolutional kernel width = number of previous timesteps that influence gate values for a given
                   timestep
  * `mode`: Can be "f", "fo", or "ifo". These correspond to the QRNN variants with the same names as described in [1]
  * `zoneout`: Type of recurrent dropout. Probability of randomly setting a recurrent gate to 1, i.e. copying the 
               previous hidden state to the current timestep without modification

## Experiments
In order to check whether the implementation is correct, a QRNN language model is compared to a LSTM language model
on the Wikitext-2v1 dataset.
However, this comparison uses a truncated vocabulary (top 10K most frequent tokens) and treats each sentence as 
input sequence.
See the accompanying jupyter notebook for data preprocessing and `experiment.py` for exact hyperparameters.
The only purpose of this comparison is to show the implementation works as expected, not to reach good performance.

Results are in the following table:

Model| Perplexity |
-----| -----------|
QRNN | xx.xx      |
LSTM | xx.xx      |


## References

[1]: Bradbury, James, et al. "Quasi-recurrent neural networks." arXiv preprint arXiv:1611.01576 (2016). 
