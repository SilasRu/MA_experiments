import numpy as np, torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class DeepLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, recurrent_dropout, use_orthnormal_init=True, fix_mask=True, use_cuda=False):
        super(DeepLSTM, self).__init__()
        self.fix_mask = fix_mask
        self.use_cuda = use_cuda
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.recurrent_dropout = recurrent_dropout
        self.lstms = nn.ModuleList([None] * self.num_layers)
        self.highway_gate_input = nn.ModuleList([None] * self.num_layers)
        self.highway_gate_state = nn.ModuleList([nn.Linear(hidden_size, hidden_size)] * self.num_layers)
        self.highway_linear_input = nn.ModuleList([None] * self.num_layers)
        for l in range(self.num_layers):
            input_dim = input_size if l == 0 else hidden_size
            self.lstms[l] = nn.LSTMCell(input_size=input_dim, hidden_size=hidden_size)
            self.highway_gate_input[l] = nn.Linear(input_dim, hidden_size)
            self.highway_linear_input[l] = nn.Linear(input_dim, hidden_size, bias=False)

        for l in range(self.num_layers):
            if use_orthnormal_init:
                init.orthogonal_(self.lstms[l].weight_ih)
                init.orthogonal_(self.lstms[l].weight_hh)
                init.orthogonal_(self.highway_gate_input[l].weight.data)
                init.orthogonal_(self.highway_gate_state[l].weight.data)
                init.orthogonal_(self.highway_linear_input[l].weight.data)
            else:
                init_weight_value = 6.0
                init.xavier_normal_((self.lstms[l].weight_ih), gain=(np.sqrt(init_weight_value)))
                init.xavier_normal_((self.lstms[l].weight_hh), gain=(np.sqrt(init_weight_value)))
                init.xavier_normal_((self.highway_gate_input[l].weight.data), gain=(np.sqrt(init_weight_value)))
                init.xavier_normal_((self.highway_gate_state[l].weight.data), gain=(np.sqrt(init_weight_value)))
                init.xavier_normal_((self.highway_linear_input[l].weight.data), gain=(np.sqrt(init_weight_value)))

    def init_hidden(self, batch_size, hidden_size):
        if self.use_cuda:
            return (
             torch.zeros(batch_size, hidden_size).cuda(),
             torch.zeros(batch_size, hidden_size).cuda())
        return (torch.zeros(batch_size, hidden_size),
         torch.zeros(batch_size, hidden_size))

    def forward(self, inputs, input_masks, Train):
        """
            inputs:       [[seq_len, batch, Co * kernel_sizes], n_layer * [None]] (list)
            input_masks:  [[seq_len, batch, Co * kernel_sizes], n_layer * [None]] (list)
        """
        batch_size, seq_len = inputs[0].size(1), inputs[0].size(0)
        self.inputs = inputs
        self.input_masks = input_masks
        if self.fix_mask:
            self.output_dropout_layers = [
             None] * self.num_layers
            for l in range(self.num_layers):
                binary_mask = torch.rand((batch_size, self.hidden_size)) > self.recurrent_dropout
                self.output_dropout_layers[l] = binary_mask.float().div(1.0 - self.recurrent_dropout)
                if self.use_cuda:
                    self.output_dropout_layers[l] = self.output_dropout_layers[l].cuda()

        for l in range(self.num_layers):
            h, c = self.init_hidden(batch_size, self.hidden_size)
            outputs_list = []
            for t in range(len(self.inputs[l])):
                x = self.inputs[l][t]
                m = self.input_masks[l][t].float()
                h_temp, c_temp = self.lstms[l].forward(x, (h, c))
                r = torch.sigmoid(self.highway_gate_input[l](x) + self.highway_gate_state[l](h))
                lx = self.highway_linear_input[l](x)
                h_temp = r * h_temp + (1 - r) * lx
                if Train:
                    if self.fix_mask:
                        h_temp = self.output_dropout_layers[l] * h_temp
                    else:
                        h_temp = F.dropout(h_temp, p=(self.recurrent_dropout))
                h = m * h_temp + (1 - m) * h
                c = m * c_temp + (1 - m) * c
                outputs_list.append(h)

            outputs = torch.stack(outputs_list, 0)
            self.inputs[l + 1] = DeepLSTM.flip(outputs, 0)
            self.input_masks[l + 1] = DeepLSTM.flip(self.input_masks[l], 0)

        self.output_state = self.inputs
        self.output_state = self.output_state[(-1)].transpose(0, 1)
        assert self.output_state.size() == (batch_size, seq_len, self.hidden_size)
        return self.output_state

    @staticmethod
    def flip(x, dim):
        xsize = x.size()
        dim = x.dim() + dim if dim < 0 else dim
        x = x.contiguous()
        x = (x.view)(*(-1, ), *xsize[dim:]).contiguous()
        x = x.view(x.size(0), x.size(1), -1)[:,
         getattr(torch.arange(x.size(1) - 1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
        return x.view(xsize)