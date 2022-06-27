import pandas as pd
import ast
from rouge_metric import PyRouge
from statistics import mean
rouge1 = PyRouge(rouge_n=(1))


def full_evaluate():
    df_ref = pd.read_csv('/Users/silas.rudolf/projects/School/MA/experiments/data/keyword-eval/reference.csv')
    predictors = ['cnn_14000', 'cnn_35000', 'xsum_10000', 'xsum_32000']
    metrics = ['rouge_1_r', 'rouge_1_p', 'rouge_1_f']
    scores = {'filename': []}
    for predictor in predictors:
        for metric in metrics:
            scores[f'{predictor}_{metric}'] = []

    for i,r in df_ref.iterrows():
        filename = f"{r['file_id']}_{i}"
        print(f'processing - {filename}')
        try:
            path = f'/Users/silas.rudolf/projects/School/MA/experiments/data/keyphrase-eval/full/{filename}.csv'
            pred_df = pd.read_csv(path)
        except:
            print(f'Could not read file {filename} - continuing')
            continue
        scores['filename'].append(filename)
        true = r['summary']
        
        for predictor in predictors:
            batch_scores = {'r': [], 'p': [], 'f': []}
            utterance = ast.literal_eval(pred_df[predictor][0])
            score1 = rouge1.evaluate_tokenized(utterance, [[true]])
            for key,val in score1['rouge-1'].items():
                    batch_scores[key].append(val)
            scores[f'{predictor}_rouge_1_r'].append(max(batch_scores['r']))
            scores[f'{predictor}_rouge_1_p'].append(max(batch_scores['p']))
            scores[f'{predictor}_rouge_1_f'].append(max(batch_scores['f']))
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(f'/Users/silas.rudolf/projects/School/MA/experiments/data/keyphrase-eval/full_scores.csv', index=False)


def batch_evaluate():
    df_ref = pd.read_csv('/Users/silas.rudolf/projects/School/MA/experiments/data/keyword-eval/reference.csv')
    predictors = ['cnn_14000', 'cnn_35000', 'xsum_10000', 'xsum_32000']
    metrics = ['rouge_1_mean_r', 'rouge_1_mean_p', 'rouge_1_mean_f', 'rouge_1_max_r', 'rouge_1_max_p', 'rouge_1_max_f']
    scores = {'filename': []}
    for predictor in predictors:
        for metric in metrics:
            scores[f'{predictor}_{metric}'] = []

    for i,r in df_ref.iterrows():
        filename = f"{r['file_id']}_{i}"
        print(f'processing - {filename}')
        try:
            path = f'/Users/silas.rudolf/projects/School/MA/experiments/data/keyphrase-eval/batch/{filename}.csv'
            pred_df = pd.read_csv(path)
        except:
            print(f'Could not read file {filename} - continuing')
            continue
        scores['filename'].append(filename)
        true = r['summary']
        
        for predictor in predictors:
            pred_strings = ast.literal_eval(pred_df[predictor][0])
            batch_scores = {'r': [], 'p': [], 'f': []}
            for utterance in pred_strings:
                score1 = rouge1.evaluate_tokenized(utterance, [[true]])
                for key,val in score1['rouge-1'].items():
                    batch_scores[key].append(val)
            scores[f'{predictor}_rouge_1_mean_r'].append(mean(batch_scores['r']))
            scores[f'{predictor}_rouge_1_mean_p'].append(mean(batch_scores['p']))
            scores[f'{predictor}_rouge_1_mean_f'].append(mean(batch_scores['f']))
            scores[f'{predictor}_rouge_1_max_r'].append(max(batch_scores['r']))
            scores[f'{predictor}_rouge_1_max_p'].append(max(batch_scores['p']))
            scores[f'{predictor}_rouge_1_max_f'].append(max(batch_scores['f']))

    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(f'/Users/silas.rudolf/projects/School/MA/experiments/data/keyphrase-eval/batch_scores.csv', index=False)

def main():
    batch_evaluate()
    full_evaluate()


main()