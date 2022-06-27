import sys
sys.path.insert(0, '/Users/silas.rudolf/projects/School/MA/experiments/models/keyword_gen')
import pandas as pd
import keyword_evaluator
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def main():
    tokens_to_remove = stopwords.words('english')
    tokens_to_remove.extend([' ', '.', ',', "'", '-', '_', '–', 'γ', ':', 'the', 'and', 'Ġ', '<s>', ')', '('])
    tokens_to_remove = list(set(tokens_to_remove))

    evaluator = keyword_evaluator.Evaluator()
    df_ref = pd.read_csv('/Users/silas.rudolf/projects/School/MA/experiments/data/keyword-eval/reference.csv')

    predictors = ['attention_max', 'attention_mean', 'attention_sum',
              'keybert_default', 'keybert_mmr', 'keybert_maxsum', 'textrank',
              'topicrank', 'positionrank', 'rake', 'yake', 'frake']
    metrics = ['precision@2', 'precision@3', 'precision@5', 'rouge_1_r', 'rouge_1_p', 'rouge_1_f']
    scores = {'filename': []}
    for predictor in predictors:
        for metric in metrics:
            scores[f'{predictor}_{metric}'] = []

    for i, r in df_ref.iterrows():
        filename = f"{r['file_id']}_{i}"
        print(f'processing - {filename}')
        try:
            path = f'/Users/silas.rudolf/projects/School/MA/experiments/data/keyword-eval/processed/{filename}.csv'
            pred_df = pd.read_csv(path)
        except:
            print(f'Could not read file {filename} - continuing')
            continue

        true = [word for word in r['summary'].split(' ') if word not in tokens_to_remove]
        scores['filename'].append(filename)

        for predictor in pred_df.columns:
            pred = [word for word in pred_df[predictor] if word not in tokens_to_remove]
            scores_for_predictor = evaluator.evaluate(true=true, pred=pred)

            for metric in metrics:
                if metric in scores_for_predictor:
                    scores[f'{predictor}_{metric}'].append(scores_for_predictor[metric])
                else:
                    scores[f'{predictor}_{metric}'].append(None)


    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(f'/Users/silas.rudolf/projects/School/MA/experiments/data/keyword-eval/scores.csv', index=False)


main()