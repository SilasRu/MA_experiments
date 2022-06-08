import json, torch, pickle as pkl
from os.path import join
from pytorch_pretrained_bert.tokenization import BertTokenizer
from .summarizer import Summarizer
from .utils import pad_batch_tensorize
PAD = 0
UNK = 1
START = 2
END = 3

class Extractor(object):

    def __init__(self, ext_dir, ext_ckpt, emb_type, max_ext=6, cuda=True):
        ext_meta = json.load(open(join(ext_dir, 'meta.json')))
        ext_args = ext_meta['model_args']
        extractor = Summarizer(**ext_args)
        extractor.load_state_dict(ext_ckpt)
        self._tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self._emb_type = emb_type
        self._device = torch.device('cuda' if cuda else 'cpu')
        self._net = extractor.to(self._device)
        self._max_ext = max_ext

    def __call__(self, raw_article_sents):
        self._net.eval()
        n_art = len(raw_article_sents)
        articles = [self._tokenizer.convert_tokens_to_ids(sentence) for sentence in raw_article_sents]
        article = pad_batch_tensorize(articles, PAD, cuda=False).to(self._device)
        indices = self._net.extract([article], k=(min(n_art, self._max_ext)))
        return indices