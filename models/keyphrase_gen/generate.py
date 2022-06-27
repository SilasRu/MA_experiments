import sys
import os
import pandas as pd
sys.path.insert(0, '/Users/silas.rudolf/projects/School/MA/experiments/models/keyphrase_gen')
from keyphrase_generator import KeyphraseGenerator

def main():
    checkpoint_path = '/Users/silas.rudolf/projects/School/MA/experiments/data/checkpoints'
    df = pd.read_csv('/Users/silas.rudolf/projects/School/MA/experiments/data/keyword-eval/reference.csv')
    generator_cnn_14000 = KeyphraseGenerator(os.path.join(checkpoint_path, 'checkpoint-CNN-keyphrase-14000'))
    generator_cnn_35000 = KeyphraseGenerator(os.path.join(checkpoint_path, 'checkpoint-CNN-keyphrase-35000'))
    generator_xsum_10000 = KeyphraseGenerator(os.path.join(checkpoint_path, 'checkpoint-keyphrase-10000'))
    generator_xsum_32000 = KeyphraseGenerator(os.path.join(checkpoint_path, 'checkpoint-keyphrase-32000'))

    for i,r in df.iterrows():
        print(f'Processing {i} / {len(df)} : file {r["file_id"]}_{i}')
        if f"{r['file_id']}_{i}.csv" in os.listdir('/Users/silas.rudolf/projects/School/MA/experiments/data/keyphrase-eval/batch'):
            print(f"already processed {r['file_id']}_{i}")
            continue
        batches = generator_cnn_14000._chunk_text(r['text'])
        batch_preds = {
            'cnn_14000': [],
            'cnn_35000': [],
            'xsum_10000': [],
            'xsum_32000': []
        }
        full_preds = {
            'cnn_14000': [],
            'cnn_35000': [],
            'xsum_10000': [],
            'xsum_32000': []
        }
        batch_preds['cnn_14000'].append(generator_cnn_14000.extract_keyphrases(batches))
        batch_preds['cnn_35000'].append(generator_cnn_35000.extract_keyphrases(batches))
        batch_preds['xsum_10000'].append(generator_xsum_10000.extract_keyphrases(batches))
        batch_preds['xsum_32000'].append(generator_xsum_32000.extract_keyphrases(batches))

        full_preds['cnn_14000'].append(generator_cnn_14000.extract_keyphrases([r['text']])[0])
        full_preds['cnn_35000'].append(generator_cnn_35000.extract_keyphrases([r['text']])[0])
        full_preds['xsum_10000'].append(generator_xsum_10000.extract_keyphrases([r['text']])[0])
        full_preds['xsum_32000'].append(generator_xsum_32000.extract_keyphrases([r['text']])[0])

        batch_df = pd.DataFrame(batch_preds)
        batch_df.to_csv(f'/Users/silas.rudolf/projects/School/MA/experiments/data/keyphrase-eval/batch/{r["file_id"]}_{i}.csv', index=False)
        
        full_df = pd.DataFrame(full_preds)
        full_df.to_csv(f'/Users/silas.rudolf/projects/School/MA/experiments/data/keyphrase-eval/full/{r["file_id"]}_{i}.csv', index=False)


main()