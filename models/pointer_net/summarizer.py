import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from pytorch_pretrained_bert.modeling import BertModel
from .DeepLSTM import DeepLSTM
from .LSTMPointerNet import LSTMPointerNet
from .utils import len_mask
from .rnn import lstm_encoder
MAX_ARTICLE_LEN = 512
INI = 1e-2

class ConvSentEncoder(nn.Module):
    __doc__ = '\n    Convolutional word-level sentence encoder\n    w/ max-over-time pooling, [3, 4, 5] kernel sizes, ReLU activation\n    '

    def __init__(self, vocab_size, emb_dim, n_hidden, dropout, emb_type):
        super().__init__()
        self._emb_type = emb_type
        self._convs = nn.ModuleList([nn.Conv1d(emb_dim, n_hidden, i) for i in range(3, 6)])
        self._dropout = dropout
        self._grad_handle = None

    def forward(self, input_):
        emb_input = input_
        conv_in = F.dropout((emb_input.transpose(1, 2)), (self._dropout), training=(self.training))
        output = torch.cat([F.relu(conv(conv_in)).max(dim=2)[0] for conv in self._convs], dim=1)
        return output

    def set_embedding(self, embedding):
        """embedding is the weight matrix"""
        assert self._embedding.weight.size() == embedding.size()
        self._embedding.weight.data.copy_(embedding)

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, n_hidden, n_layer, dropout, bidirectional):
        super().__init__()
        self._init_h = nn.Parameter(
            torch.Tensor(n_layer*(2 if bidirectional else 1), n_hidden))
        self._init_c = nn.Parameter(
            torch.Tensor(n_layer*(2 if bidirectional else 1), n_hidden))
        init.uniform_(self._init_h, -INI, INI)
        init.uniform_(self._init_c, -INI, INI)
        self._lstm = nn.LSTM(input_dim, n_hidden, n_layer,
                             dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_, in_lens=None):
        """ [batch_size, max_num_sent, input_dim] Tensor"""
        size = (self._init_h.size(0), input_.size(0), self._init_h.size(1))
        init_states = (self._init_h.unsqueeze(1).expand(*size),
                       self._init_c.unsqueeze(1).expand(*size))
        lstm_out, _ = lstm_encoder(input_, self._lstm, in_lens, init_states)
        return lstm_out.transpose(0, 1)

    @property
    def input_size(self):
        return self._lstm.input_size

    @property
    def hidden_size(self):
        return self._lstm.hidden_size

    @property
    def num_layers(self):
        return self._lstm.num_layers

    @property
    def bidirectional(self):
        return self._lstm.bidirectional


class Summarizer(nn.Module):
    __doc__ = ' Different encoder/decoder/embedding type '

    def __init__(self, encoder, decoder, emb_type, emb_dim, vocab_size, conv_hidden, encoder_hidden, encoder_layer, isTrain=True, n_hop=1, dropout=0.0):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._emb_type = emb_type
        self._sent_enc = ConvSentEncoder(vocab_size, emb_dim, conv_hidden, dropout, emb_type)
        if emb_type == 'BERT':
            self._bert = BertModel.from_pretrained('bert-large-uncased')
            self._bert.eval()
            for p in self._bert.parameters():
                p.requires_grad = False

            self._bert_w = nn.Linear(4096, emb_dim)
                # Sentence Encoder
        if encoder == 'BiLSTM':
            enc_out_dim = encoder_hidden * 2 # bidirectional
            self._art_enc = LSTMEncoder(
                3*conv_hidden, encoder_hidden, encoder_layer,
                dropout=dropout, bidirectional=True
            )
        if encoder == 'DeepLSTM':
            enc_out_dim = encoder_hidden
            self._isTrain = isTrain
            self._art_enc = DeepLSTM(3 * conv_hidden, encoder_hidden, encoder_layer, 0.1)
        decoder_hidden = encoder_hidden
        decoder_layer = encoder_layer
        self._extractor = LSTMPointerNet(enc_out_dim, decoder_hidden, decoder_layer, dropout, n_hop)

    def forward(self, article_sents, sent_nums, target):
        enc_out = self._encode(article_sents, sent_nums)
        if self._decoder == 'PN':
            bs, nt = target.size()
            d = enc_out.size(2)
            ptr_in = torch.gather(enc_out,
              dim=1, index=(target.unsqueeze(2).expand(bs, nt, d)))
            output = self._extractor(enc_out, sent_nums, ptr_in)
        else:
            bs, seq_len, d = enc_out.size()
            output = self._ws(enc_out)
            assert output.size() == (bs, seq_len, 2)
        return output

    def extract(self, article_sents, sent_nums=None, k=4):
        enc_out = self._encode(article_sents, sent_nums)
        if self._decoder == 'PN':
            extract = self._extractor.extract(enc_out, sent_nums, k)
        else:
            seq_len = enc_out.size(1)
            output = self._ws(enc_out)
            assert output.size() == (1, seq_len, 2)
            _, indices = output[:, :, 1].sort(descending=True)
            extract = []
            for i in range(k):
                extract.append(indices[0][i].item())

        return extract

    def _encode(self, article_sents, sent_nums):
        hidden_size = self._art_enc.input_size
        if sent_nums is None:
            enc_sent = self._article_encode(article=(article_sents[0]), device=(article_sents[0].device)).unsqueeze(0)
        else:
            max_n = max(sent_nums)
            enc_sents = [self._article_encode(article=article, device=(article.device)) for article in article_sents]

            def zero(n, device):
                z = torch.zeros(n, hidden_size).to(device)
                return z

            enc_sent = torch.stack([torch.cat([s, zero(max_n - n, s.get_device())], dim=0) if n != max_n else s for s, n in zip(enc_sents, sent_nums)],
              dim=0)

        if self._encoder == 'BiLSTM':
            output = self._art_enc(enc_sent, sent_nums)

        if self._encoder == 'DeepLSTM':
            batch_size, seq_len = enc_sent.size(0), enc_sent.size(1)
            inputs = [enc_sent.transpose(0, 1)]
            if sent_nums != None:
                inputs_mask = [
                 len_mask(sent_nums, enc_sent.get_device()).transpose(0, 1).unsqueeze(-1)]
            else:
                inputs_mask = [
                 torch.ones(seq_len, batch_size, 1)]
            for _ in range(self._art_enc.num_layers):
                inputs.append([None])
                inputs_mask.append([None])

            assert inputs[0].size() == (seq_len, batch_size, hidden_size)
            assert inputs_mask[0].size() == (seq_len, batch_size, 1)
            output = self._art_enc(inputs, inputs_mask, self._isTrain)
        return output

    def _article_encode(self, article, device, pad_idx=0):
        sent_num, sent_len = article.size()
        tokens_id = [101]
        for i in range(sent_num):
            for j in range(sent_len):
                if article[i][j] != pad_idx:
                    tokens_id.append(article[i][j])
                else:
                    break

        tokens_id.append(102)
        input_mask = [1] * len(tokens_id)
        total_len = len(tokens_id) - 2
        while len(tokens_id) < MAX_ARTICLE_LEN:
            tokens_id.append(0)
            input_mask.append(0)

        assert len(tokens_id) == MAX_ARTICLE_LEN
        assert len(input_mask) == MAX_ARTICLE_LEN
        input_ids = torch.LongTensor(tokens_id).unsqueeze(0).to(device)
        input_mask = torch.LongTensor(input_mask).unsqueeze(0).to(device)
        out, _ = self._bert(input_ids, token_type_ids=None, attention_mask=input_mask)
        out = torch.cat([out[(-1)], out[(-2)], out[(-3)], out[(-4)]], dim=(-1))
        assert out.size() == (1, MAX_ARTICLE_LEN, 4096)
        emb_out = self._bert_w(out).squeeze(0)
        emb_dim = emb_out.size(-1)
        emb_input = torch.zeros(sent_num, sent_len, emb_dim).to(device)
        cur_idx = 1
        for i in range(sent_num):
            for j in range(sent_len):
                if article[i][j] != pad_idx:
                    emb_input[i][j] = emb_out[cur_idx]
                    cur_idx += 1
                else:
                    break

        assert cur_idx - 1 == total_len
        cnn_out = self._sent_enc(emb_input)
        assert cnn_out.size() == (sent_num, 300)
        return cnn_out

    def set_embedding(self, embedding):
        self._sent_enc.set_embedding(embedding)