import torch
MAX_ARTICLE_LEN = 512
from pytorch_pretrained_bert.tokenization import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

def reorder_sequence(sequence_emb, order, batch_first=False):
    """
    sequence_emb: [T, B, D] if not batch_first
    order: list of sequence length
    """
    batch_dim = 0 if batch_first else 1
    assert len(order) == sequence_emb.size()[batch_dim]
    order = torch.LongTensor(order).to(sequence_emb.get_device())
    sorted_ = sequence_emb.index_select(index=order, dim=batch_dim)
    return sorted_


def reorder_lstm_states(lstm_states, order):
    """
    lstm_states: (H, C) of tensor [layer, batch, hidden]
    order: list of sequence length
    """
    assert isinstance(lstm_states, tuple)
    assert len(lstm_states) == 2
    assert lstm_states[0].size() == lstm_states[1].size()
    assert len(order) == lstm_states[0].size()[1]
    order = torch.LongTensor(order).to(lstm_states[0].get_device())
    sorted_states = (lstm_states[0].index_select(index=order, dim=1),
     lstm_states[1].index_select(index=order, dim=1))
    return sorted_states


def len_mask(lens, device):
    """ users are resposible for shaping
    Return: tensor_type [B, T]
    """
    max_len = max(lens)
    batch_size = len(lens)
    mask = torch.ByteTensor(batch_size, max_len).to(device)
    mask.fill_(0)
    for i, l in enumerate(lens):
        mask[i, :l].fill_(1)

    return mask


def pad_batch_tensorize(inputs, pad, cuda=True):
    """pad_batch_tensorize

    :param inputs: List of size B containing torch tensors of shape [T, ...]
    :type inputs: List[np.ndarray]
    :rtype: TorchTensor of size (B, T, ...)
    """
    tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    batch_size = len(inputs)
    max_len = max((len(ids) for ids in inputs))
    tensor_shape = (batch_size, max_len)
    tensor = tensor_type(*tensor_shape)
    tensor.fill_(pad)
    for i, ids in enumerate(inputs):
        tensor[i, :len(ids)] = tensor_type(ids)

    return tensor


def tokenize(max_len, texts):
    truncated_article = []
    left = MAX_ARTICLE_LEN - 2
    for sentence in texts:
        tokens = tokenizer.tokenize(sentence)
        tokens = tokens[:max_len]
        if left >= len(tokens):
            truncated_article.append(tokens)
            left -= len(tokens)
        else:
            truncated_article.append(tokens[0:left])
            break

    return truncated_article