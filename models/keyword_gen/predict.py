import sys
sys.path.insert(0, '/Users/silas.rudolf/projects/School/MA/experiments/models/keyword_gen')
import pandas as pd
import os
import keyword_generator as kwg


def main():
    df = pd.read_csv('/Users/silas.rudolf/projects/School/MA/experiments/data/keyword-eval/reference.csv')

    kw = kwg.KeywordGenerator()
    attention_kwg = kwg.Attention(model_path='bert-base-uncased')
    bert_kwg = kwg.KeyBert()
    textrank_kwg = kwg.TextRank()
    topicrank_kwg = kwg.TopicRank()
    positionrank_kwg = kwg.PositionRank()
    rake_kwg = kwg.RakeNltk()
    yake_kwg = kwg.Yake()
    frake_kwg = kwg.Frake()

    for i,r in df.iterrows():
        print(f'Processing {i} / {len(df)} : file {r["file_id"]}_{i}')
        if f"{r['file_id']}_{i}.csv" in os.listdir('/Users/silas.rudolf/projects/School/MA/experiments/data/keyword-eval'):
            print(f"already processed {r['file_id']}_{i}")
            continue
        batches = kw.split_text(r['text'])
        batch_preds = []
        try:
            for batch in batches:
                kws = {}
                kws.update(attention_kwg.extract_keywords(batch)) 
                kws.update(bert_kwg.extract_keywords(batch))
                kws.update(textrank_kwg.extract_keywords(batch))
                kws.update(topicrank_kwg.extract_keywords(batch))
                kws.update(positionrank_kwg.extract_keywords(batch))
                kws.update(rake_kwg.extract_keywords(batch))
                kws.update(yake_kwg.extract_keywords(batch))
                kws.update(frake_kwg.extract_keywords(batch))
                batch_preds.append(kws)
            merged = kw.merge_batch_keywords(batch_preds)
            batch_df = pd.DataFrame(merged)
            batch_df.to_csv(f'/Users/silas.rudolf/projects/School/MA/experiments/data/keyword-eval/{r["file_id"]}_{i}.csv', index=False)
        except:
            print(f"skipping {r['file_id']}")



main()
