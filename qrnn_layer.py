import torch
import torch.nn as nn
import torch.nn.functional as F


class QRNNLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, kernel_size: int,
                 dropout: float):
        super(QRNNLayer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dropout = dropout
        
        self.z_conv = nn.Conv1d(input_size, hidden_size, kernel_size, padding="same")
        self.f_conv = nn.Conv1d(input_size, hidden_size, kernel_size, padding="same")
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs = shape: [batch x timesteps x features]
        batch, timesteps, _ = inputs.shape
        
        # Apply convolutions
        inputs = inputs.transpose(1, 2)
        f_raw = self.f_conv(inputs).transpose(1, 2)
        z_raw = self.z_conv(inputs).transpose(1, 2)
        
        # Get log values of activations
        log_z = torch.log(torch.nn.functional.softplus(raw_z))
        log_f = torch.nn.functional.logsigmoid(raw_f)
        log_one_minus_f = torch.nn.functional.logsigmoid(-raw_f)
        
        # Precalculate recurrent gate values
        recurrent_gates = log_f[:, 1:, :]
        recurrent_gates_cumsum = torch.cumsum(recurrent_gates, dim=1)
        recurrent_gates = recurrent_gates - recurrent_gates_cumsum + recurrent_gates_cumsum[:, -1:, :]
        padding = torch.zeros([batch, 1, self.hidden_size], device=recurrent_gates.device)
        recurrent_gates = torch.cat([recurrent_gates, padding], dim=1)
        
        # Calculate recursion
        log_hidden = torch.logcumsumexp(log_z + log_one_minus_f + recurrent_gates, axis=1)
        hidden = torch.exp(log_hidden - recurrent_gates)
        
        return hidden
