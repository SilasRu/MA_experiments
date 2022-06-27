from nltk.stem import SnowballStemmer
from rouge_metric import PyRouge
stemmer = SnowballStemmer('english')
rouge = PyRouge(rouge_n=(1))

class Evaluator:
    def __init__(self) -> None:
        self.model = 'Evaluator'

    def _stem(self, true, pred):
        true_stems = [stemmer.stem(str(i)) for i in true]
        pred_stems = [stemmer.stem(str(i)) for i in pred]
        return true_stems, pred_stems

    def _precision_k(self, true, pred, k):
        try:
            assert k <= len(true) and k <= len(pred)
        except:
            # print(f'k {k} is smaller than true: {len(true)} or pred: {len(pred)}')
            return 
        true_stems, pred_stems = self._stem(true, pred)

        true_pos = 0
        seen = set()
        
        for i in range(k):
            if pred[i] in true_stems and pred_stems[i] not in seen:
                seen.add(pred[i])
                true_pos += 1
        score = round(true_pos / k, 2)
        return {f'precision@{k}': score}
    
    def _rogue(self, true, pred):
        true_stems, pred_stems = self._stem(true, pred)
        if len(true_stems) < len(pred_stems):
            pred_stems = pred_stems[:len(true_stems)]
        if len(pred_stems) < len(true_stems):
            true_stems = true_stems[:len(pred_stems)]
        
        hyps = true_stems
        refs = [pred_stems for _ in pred_stems]
        print(hyps)
        print(refs)
        score = rouge.evaluate_tokenized(hyps, refs)
        score_rounded = {
            'rouge_1_r': round(score['rouge-1']['r'], 2),
            'rouge_1_p': round(score['rouge-1']['p'], 2),
            'rouge_1_f': round(score['rouge-1']['f'], 2)
        }
        return score_rounded

    
    def evaluate(self, true, pred):
        if len(true) < len(pred):
            # print(f'Trimming pred to len {len(true)}')
            pred = pred[:len(true)]
        prec_2 = self._precision_k(true, pred, 2)
        prec_3 = self._precision_k(true, pred, 3)
        prec_5 = self._precision_k(true, pred, 5)
        rouge = self._rogue(true, pred)
        scores = {}
        if type(prec_2) == dict: scores.update(prec_2)
        if type(prec_3) == dict: scores.update(prec_3)
        if type(prec_5) == dict: scores.update(prec_5)
        if type(rouge) == dict: scores.update(rouge)
        return scores

