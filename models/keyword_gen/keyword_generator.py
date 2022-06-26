import nltk
import kex
import yake
import FRAKE.FRAKE as frk
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, utils
from keybert import KeyBERT
from rake_nltk import Rake
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
nltk.download('stopwords')
utils.logging.set_verbosity_error()  # Suppress standard warnings
stemmer = SnowballStemmer('english')


class KeywordGenerator:
    def __init__(self) -> None:
        pass

    def clean_data(self):
        pass

    def extract_keywords(self):
        pass

    def split_text(self, text):
        split_len = 1000
        batches = [text[i: i + split_len] for i in range(0, len(text), split_len)]
        return batches

    def merge_batch_keywords(self, batches):
        top_n = {key: [] for key in batches[0]}
        for batch in batches:
            for key in batch.keys():
                try:
                    i = 0
                    while batch[key][i] in top_n[key] and i <= len(batch[key]):
                        i += 1
                    top_n[key].append(batch[key][i])
                except:
                    print(f'could not process {key}')
                    top_n[key].append('None')
        return top_n

    def _deduplicate_words(self, words: list) -> list:
        '''Checks for duplicate keywords based on the word stem and starting 4 letters.

        Returns:
            words: unique words
        '''
        stemmed_words_index = []
        stemmed_words = set()

        for i, word in enumerate(words):
            stem = stemmer.stem(word)[:4]
            if stem not in stemmed_words:
                stemmed_words.add(stem)
                stemmed_words_index.append(i)

        return np.array(words)[stemmed_words_index]


class KeyBert(KeywordGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.model = KeyBERT()

    def extract_keywords(self, text):
        super().extract_keywords()
        kws = self.model.extract_keywords(text, keyphrase_ngram_range=(1, 1), top_n=8)
        kws_mmr = self.model.extract_keywords(text, keyphrase_ngram_range=(1, 1), use_mmr=True, top_n=8)
        kws_maxsum = self.model.extract_keywords(text, keyphrase_ngram_range=(1, 1), use_maxsum=True, top_n=8)

        kws = {
            'keybert_default': self._deduplicate_words([word for word, score in kws])[:5],
            'keybert_mmr': self._deduplicate_words([word for word, score in kws_mmr])[:5],
            'keybert_maxsum': self._deduplicate_words([word for word, score in kws_maxsum])[:5]
        }
        return kws


class RakeNltk(KeywordGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.model = Rake()

    def extract_keywords(self, text):
        self.model.extract_keywords_from_text(text)
        extracted_kws = self.model.get_ranked_phrases()
        kws = extracted_kws[:10]
        return {
            'rake': self._deduplicate_words(kws)[:5]
        }


class Frake(KeywordGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.model = frk.KeywordExtractor(lang='en', Number_of_keywords=10)

    def extract_keywords(self, text):
        extracted_kws = self.model.extract_keywords(text)
        kws = list(extracted_kws.keys())
        return {
            'frake': self._deduplicate_words(kws)[:5]
        }


class Yake(KeywordGenerator):
    def __init__(self) -> None:
        super().__init__()
        language = "en"
        max_ngram_size = 2
        deduplication_thresold = 0.5
        deduplication_algo = 'seqm'
        windowSize = 1
        numOfKeywords = 10
        self.model = yake.KeywordExtractor(
            lan=language,
            n=max_ngram_size,
            dedupLim=deduplication_thresold,
            dedupFunc=deduplication_algo,
            windowsSize=windowSize,
            top=numOfKeywords, features=None
        )

    def extract_keywords(self, text):
        extracted_kws = self.model.extract_keywords(text)
        kws = []
        for word in extracted_kws:
            kws.append(word[0])
        return {
            'yake': self._deduplicate_words(kws)[:5]
        }


class TextRank(KeywordGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.model = kex.TextRank()

    def extract_keywords(self, text):
        extracted_kws = self.model.get_keywords(text, n_keywords=10)
        kws = []
        for phrase in extracted_kws:
            kws.append(phrase['raw'][0])
        return {
            'textrank': self._deduplicate_words(kws)[:5]
        }


class TopicRank(KeywordGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.model = kex.TopicRank()

    def extract_keywords(self, text):
        extracted_kws = self.model.get_keywords(text, n_keywords=10)
        kws = []
        for phrase in extracted_kws:
            kws.append(phrase['raw'][0])
        return {
            'topicrank': self._deduplicate_words(kws)[:5]
        }


class PositionRank(KeywordGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.model = kex.PositionRank()

    def extract_keywords(self, text):
        extracted_kws = self.model.get_keywords(text, n_keywords=10)
        kws = []
        for phrase in extracted_kws:
            kws.append(phrase['raw'][0])
        return {
            'positionrank': self._deduplicate_words(kws)[:5]
        }


class Attention(KeywordGenerator):
    def __init__(self, model_path) -> None:
        super().__init__()
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, output_attentions=True)
        self.convolution_operators = ['SUM', 'MEAN', 'MAX', 'DOT']

        tokens_to_remove = stopwords.words('english')
        tokens_to_remove.extend(self.tokenizer.all_special_tokens)
        tokens_to_remove.extend(['.', ',', "'", '-', '_', '–', 'γ', ':', 'the', 'and', 'Ġ', '<s>', ')', '('])
        self.tokens_to_remove = list(set(tokens_to_remove))

    def _encode_text(self, text):
        '''encodes a string input

        Returns:
            inputs: (0,len(input))
            outputs: [hidden_state, pooler_output, attentions]
                hidden_state: [batch_size, seq_len, embedding_size]
                pooler_output: [batch_size, embedding_size]
                attentions: n_layers x [batch_size, num_heads, seq_len, seq_len]
            tokens: [seq_len] -> includes padding + sep
        '''
        inputs = self.tokenizer.encode(text, return_tensors='pt')
        # Output includes attention weights when output_attentions=True
        outputs = self.model(inputs)
        attentions = outputs[-1]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs[0])

        encodings = {
            'inputs': inputs,
            'outputs': outputs,
            'attentions': attentions,
            'tokens': tokens
        }
        return encodings

    def _get_attentions_for_layer_and_head(self, attentions, layer, attention_head):
        '''get the particular output for a particular layer and attention head

        Returns:
            attentions_for_layer: [[seq_len] x [seq_len]]
        '''
        attentions_for_layer_and_head = attentions[layer].squeeze(0)[attention_head]
        return attentions_for_layer_and_head

    def _convolute_attentions_for_heads(self, attentions, layer, operator='MEAN'):
        '''merges the attention-head outputs for one layer into one matrix

        Returns:
            attentions_for_layer: [[seq_len] x [seq_len]]
        '''
        if operator == 'MEAN':
            attentions_for_layer = attentions[layer].squeeze(0).mean(dim=0)
        elif operator == 'SUM':
            attentions_for_layer = attentions[layer].squeeze(0).sum(dim=0)
        elif operator == 'MAX':
            attentions_for_layer = attentions[layer].squeeze(0).max(dim=0)[0]
        elif operator == 'DOT':
            attentions_for_layer = attentions[layer].squeeze(0)[0]
            for head in attentions[layer].squeeze(0)[1:]:
                attentions_for_layer = attentions_for_layer @ head
        return attentions_for_layer

    def _convolute_attentions(self, attentions, head_operator, layer_operator):
        '''merges the layer outputs into one single matrix

        Returns:
            convoluted_attentions: [[seq_len] x [seq_len]]
        '''
        attentions_per_layer = []
        for i in range(len(attentions)):
            atts = self._convolute_attentions_for_heads(attentions, i, head_operator)
            attentions_per_layer.append(atts.detach().numpy())

        if layer_operator == 'MEAN':
            convoluted_attentions = np.array(attentions_per_layer).mean(axis=0)
        elif layer_operator == 'SUM':
            convoluted_attentions = np.array(attentions_per_layer).sum(axis=0)
        elif layer_operator == 'MAX':
            convoluted_attentions = np.array(attentions_per_layer).max(axis=0)
        elif layer_operator == 'DOT':
            convoluted_attentions = np.array(attentions_per_layer)[0]
            for head in attentions_per_layer[1:]:
                convoluted_attentions = convoluted_attentions @ np.array(head)
        return convoluted_attentions

    def _extract_keyword_dict(self, text):
        '''extracts keywords with all convolution operators

        Returns:
            extraction_dict: {tokens=[], [operators=[]]}
        '''
        encodings = self._encode_text(text)
        extraction_dict = {'tokens': encodings['tokens']}

        for head_operator in self.convolution_operators:
            for layer_operator in self.convolution_operators:
                for axis in [0, 1]:
                    extraction_dict[f'{head_operator}_{layer_operator}_{axis}'] = self._convolute_attentions(encodings['attentions'], head_operator, layer_operator).mean(axis=axis)

        return extraction_dict

    def _clean_and_convert_keyword_dict(self, keyword_dict):
        '''removes special tokens, separators and stopwords and converts the dict to a pandas df

        Returns:
            df: pandas df
        '''
        df = pd.DataFrame(keyword_dict)
        index_to_drop = []
        for i, r in df.iterrows():
            if r['tokens'] in self.tokens_to_remove or '#' in r['tokens']:
                index_to_drop.append(i)
        df.drop(index=index_to_drop, inplace=True)
        return df

    def _normalize_df(self, keyword_df):
        kws_normalized = keyword_df.iloc[:, 1:].divide(keyword_df.iloc[:, 1:].sum(axis=0), axis=1)
        kws_normalized['row_mean'] = kws_normalized.mean(axis=1)
        kws_normalized['row_max'] = kws_normalized.max(axis=1)
        kws_normalized['row_sum'] = kws_normalized.sum(axis=1)
        kws_normalized['tokens'] = keyword_df['tokens']
        return kws_normalized

    def _extract_ranked_keywords(self, df: pd.DataFrame, n_keywords: int) -> pd.DataFrame:
        sorted_dict = {}
        for column in df.columns[:-1]:
            sorted_dict[column] = df.sort_values(by=column, ascending=False)['tokens'][:n_keywords].values
        sorted_df = pd.DataFrame(sorted_dict)
        return sorted_df

    def extract_keywords(self, text):
        keyword_dict = self._extract_keyword_dict(text)
        keyword_df = self._clean_and_convert_keyword_dict(keyword_dict)
        normalized_df = self._normalize_df(keyword_df)
        sorted_df = self._extract_ranked_keywords(normalized_df, 20)
        kws = {
            'attention_max': self._deduplicate_words(sorted_df['row_max'])[:5],
            'attention_mean': self._deduplicate_words(sorted_df['row_mean'])[:5],
            'attention_sum': self._deduplicate_words(sorted_df['row_sum'])[:5]
        }
        return kws
