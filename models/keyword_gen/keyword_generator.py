import numpy as np
import nltk
import pandas as pd
from transformers import AutoTokenizer, AutoModel, utils
from nltk.corpus import stopwords
nltk.download('stopwords')
utils.logging.set_verbosity_error()  # Suppress standard warnings


class KeywordGenerator:
    def __init__(self, text) -> None:
        self.text = text
    

    def clean_data(self):
        pass


    def extract_keywords(self):
        pass



class AttentionKeywordGenerator(KeywordGenerator):
    def __init__(self, text, model_path) -> None:
         super().__init__(text)
         self.model_path = model_path
         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
         self.model = AutoModel.from_pretrained(model_path, output_attentions=True)
         self.convolution_operators = ['SUM', 'MEAN', 'MAX', 'DOT']

         tokens_to_remove = stopwords.words('english')
         tokens_to_remove.extend(self.tokenizer.all_special_tokens)
         tokens_to_remove.extend(['.', ',', "'", '-', ':', 'the', 'and', 'Ġ', '<s>'])
         self.tokens_to_remove = list(set(tokens_to_remove))

    def _encode_text(self):
        '''encodes a string input

        Returns:
            inputs: (0,len(input))
            outputs: [hidden_state, pooler_output, attentions]
                hidden_state: [batch_size, seq_len, embedding_size]
                pooler_output: [batch_size, embedding_size]
                attentions: n_layers x [batch_size, num_heads, seq_len, seq_len]
            tokens: [seq_len] -> includes padding + sep
        '''
        inputs = self.tokenizer.encode(self.text, return_tensors='pt')
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

    
    def _extract_keyword_dict(self):
        '''extracts keywords with all convolution operators

        Returns:
            extraction_dict: {tokens=[], [operators=[]]}
        '''
        encodings = self._encode_text()
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
        for i,r in df.iterrows():
            if r['tokens'] in self.tokens_to_remove or '#' in r['tokens']:
                index_to_drop.append(i)
        df.drop(index=index_to_drop, inplace=True)
        return df

    def _normalize_df(self, keyword_df):
        kws_normalized= keyword_df.iloc[:, 1:].divide(keyword_df.iloc[:, 1:].sum(axis=0), axis=1)
        kws_normalized['row_mean'] = kws_normalized.mean(axis=1)
        kws_normalized['row_max'] = kws_normalized.max(axis=1)
        kws_normalized['row_sum'] = kws_normalized.sum(axis=1)
        kws_normalized['tokens'] = keyword_df['tokens']
        # kws_normalized.drop_duplicates(subset ="tokens",keep = False, inplace = True)
        return kws_normalized

    def extract_keywords(self):
        keyword_dict = self._extract_keyword_dict()
        keyword_df = self._clean_and_convert_keyword_dict(keyword_dict)
        normalized_df = self._normalize_df(keyword_df)
        return normalized_df












        


    
