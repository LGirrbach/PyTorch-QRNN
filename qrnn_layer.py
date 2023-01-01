import torch
import torch.nn as nn
import torch.nn.functional as functional


class QRNNLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, kernel_size: int, mode: str = "f", zoneout: float = 0.0):
        super(QRNNLayer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.mode = mode
        self.zoneout = zoneout

        self.zoneout_distribution = torch.distributions.Bernoulli(probs=self.zoneout)
        self.pad = nn.ConstantPad1d((self.kernel_size-1, 0), value=0.0)
        self.z_conv = nn.Conv1d(input_size, hidden_size, kernel_size)
        self.f_conv = nn.Conv1d(input_size, hidden_size, kernel_size)

        if self.mode == "fo" or self.mode == "ifo":
            self.o_conv = nn.Conv1d(input_size, hidden_size, kernel_size)

        if self.mode == "ifo":
            self.i_conv = nn.Conv1d(input_size, hidden_size, kernel_size)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs = shape: [batch x timesteps x features]
        batch, timesteps, _ = inputs.shape
        
        # Apply convolutions
        inputs = inputs.transpose(1, 2)
        inputs = self.pad(inputs)
        raw_f = self.f_conv(inputs).transpose(1, 2)
        raw_z = self.z_conv(inputs).transpose(1, 2)

        if self.mode == "ifo":
            raw_i = self.i_conv(inputs).transpose(1, 2)
            log_one_minus_f = functional.logsigmoid(raw_i)
        else:
            log_one_minus_f = functional.logsigmoid(-raw_f)
        
        # Get log values of activations
        log_z = functional.logsigmoid(raw_z)  # Use sigmoid activation
        log_f = functional.logsigmoid(raw_f)

        # Optionally apply zoneout
        if self.zoneout > 0.0:
            zoneout_mask = self.zoneout_distribution.sample(sample_shape=log_f.shape).bool()
            zoneout_mask = zoneout_mask.to(log_f.device)
            log_f = torch.masked_fill(input=log_f, mask=zoneout_mask, value=0.0)
            log_one_minus_f = torch.masked_fill(input=log_one_minus_f, mask=zoneout_mask, value=-1e8)
        
        # Precalculate recurrent gate values by reverse cumsum
        recurrent_gates = log_f[:, 1:, :]
        recurrent_gates_cumsum = torch.cumsum(recurrent_gates, dim=1)
        recurrent_gates = recurrent_gates - recurrent_gates_cumsum + recurrent_gates_cumsum[:, -1:, :]
        
        # Pad last timestep
        padding = torch.zeros([batch, 1, self.hidden_size], device=recurrent_gates.device)
        recurrent_gates = torch.cat([recurrent_gates, padding], dim=1)
        
        # Calculate expanded recursion by cumsum (logcumsumexp in log space)
        log_hidden = torch.logcumsumexp(log_z + log_one_minus_f + recurrent_gates, dim=1)
        hidden = torch.exp(log_hidden - recurrent_gates)

        # Optionally multiply by output gate
        if self.mode == "fo" or self.mode == "ifo":
            o = torch.sigmoid(self.o_conv(inputs)).transpose(1, 2)
            hidden = hidden * o
        
        return hidden
