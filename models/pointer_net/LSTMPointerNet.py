import torch
from torch import nn
from torch.nn import init
import torch.nn as F
from .utils import len_mask
from .rnn import MultiLayerLSTMCells
INI = 0.01

class LSTMPointerNet(nn.Module):
    __doc__ = 'Pointer network as in Vinyals et al '

    def __init__(self, input_dim, n_hidden, n_layer, dropout, n_hop, bidirectional=True):
        super().__init__()
        self._init_h = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_c = nn.Parameter(torch.Tensor(n_layer, n_hidden))
        self._init_i = torch.Tensor(input_dim)
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        init.uniform_(self._init_i, -0.1, 0.1)
        self._lstm = nn.LSTM(input_dim,
          n_hidden, n_layer, bidirectional=False,
          dropout=dropout)
        self._lstm_cell = None
        self._attn_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._attn_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._attn_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._attn_wm)
        init.xavier_normal_(self._attn_wq)
        init.uniform_(self._attn_v, -INI, INI)
        self._hop_wm = nn.Parameter(torch.Tensor(input_dim, n_hidden))
        self._hop_wq = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self._hop_v = nn.Parameter(torch.Tensor(n_hidden))
        init.xavier_normal_(self._hop_wm)
        init.xavier_normal_(self._hop_wq)
        init.uniform_(self._hop_v, -INI, INI)
        self._n_hop = n_hop

    def forward(self, attn_mem, mem_sizes, lstm_in):
        """atten_mem: Tensor of size [batch_size, max_sent_num, input_dim]"""
        attn_feat, hop_feat, lstm_states, init_i = self._prepare(attn_mem, mem_sizes)
        lstm_in = torch.cat([init_i, lstm_in], dim=1).transpose(0, 1)
        query, final_states = self._lstm(lstm_in, lstm_states)
        query = query.transpose(0, 1)
        for _ in range(self._n_hop):
            query = LSTMPointerNet.attention(hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)

        output = LSTMPointerNet.attention_score(attn_feat, query, self._attn_v, self._attn_wq)
        return output

    def extract(self, attn_mem, mem_sizes, k):
        """extract k sentences, decode only, batch_size==1"""
        attn_feat, hop_feat, lstm_states, lstm_in = self._prepare(attn_mem, mem_sizes)
        lstm_in = lstm_in.squeeze(1)
        if self._lstm_cell is None:
            self._lstm_cell = MultiLayerLSTMCells.convert(self._lstm)
        extracts = []
        for _ in range(k):
            h, c = self._lstm_cell(lstm_in, lstm_states)
            query = h[(-1)]
            for _ in range(self._n_hop):
                query = LSTMPointerNet.attention(hop_feat, query, self._hop_v, self._hop_wq, mem_sizes)

            score = LSTMPointerNet.attention_score(attn_feat, query, self._attn_v, self._attn_wq)
            score = score.squeeze()
            for e in extracts:
                score[e] = -1000000.0

            ext = score.max(dim=0)[1].item()
            extracts.append(ext)
            lstm_states = (h, c)
            lstm_in = attn_mem[:, ext, :]

        return extracts

    def _prepare(self, attn_mem, mem_sizes):
        attn_feat = torch.matmul(attn_mem, self._attn_wm.unsqueeze(0))
        hop_feat = torch.matmul(attn_mem, self._hop_wm.unsqueeze(0))
        bs = attn_mem.size(0)
        n_l, d = self._init_h.size()
        size = (n_l, bs, d)
        lstm_states = ((self._init_h.unsqueeze(1).expand)(*size).contiguous(),
         (self._init_c.unsqueeze(1).expand)(*size).contiguous())
        d = self._init_i.size(0)
        init_i = LSTMPointerNet.max_pooling(attn_mem, mem_sizes)
        return (
         attn_feat, hop_feat, lstm_states, init_i)

    @staticmethod
    def attention_score(attention, query, v, w):
        """ unnormalized attention score"""
        sum_ = attention.unsqueeze(1) + torch.matmul(query, w.unsqueeze(0)).unsqueeze(2)
        score = torch.matmul(torch.tanh(sum_), v.unsqueeze(0).unsqueeze(1).unsqueeze(3)).squeeze(3)
        return score

    @staticmethod
    def attention(attention, query, v, w, mem_sizes):
        """ attention context vector"""
        score = LSTMPointerNet.attention_score(attention, query, v, w)
        if mem_sizes is None:
            norm_score = F.softmax(score, dim=(-1))
        else:
            mask = len_mask(mem_sizes, score.get_device()).unsqueeze(-2)
            norm_score = prob_normalize(score, mask)
        output = torch.matmul(norm_score, attention)
        return output

    @staticmethod
    def mean_pooling(attn_mem, mem_sizes):
        if mem_sizes is None:
            lens = torch.Tensor([attn_mem.size(1)]).cuda()
        else:
            lens = torch.Tensor(mem_sizes).unsqueeze(1).cuda()
        init_i = torch.sum(attn_mem, dim=1) / lens
        init_i = init_i.unsqueeze(1)
        return init_i

    @staticmethod
    def max_pooling(attn_mem, mem_sizes):
        if mem_sizes is not None:
            B, Ns = attn_mem.size(0), attn_mem.size(1)
            mask = torch.ByteTensor(B, Ns).cuda()
            mask.fill_(0)
            for i, l in enumerate(mem_sizes):
                mask[i, :l].fill_(1)

            mask = mask.unsqueeze(-1)
            attn_mem = attn_mem.masked_fill(mask == 0, -1e+18)
        init_i = attn_mem.max(dim=1, keepdim=True)[0]
        return init_i


def prob_normalize(score, mask):
    """ [(...), T]
    user should handle mask shape"""
    score = score.masked_fill(mask == 0, -1e+18)
    norm_score = F.softmax(score, dim=(-1))
    return norm_score