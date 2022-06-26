from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json



class KeyphraseGenerator:
    def __init__(self, model_path) -> None:
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)


    def preprocess_and_chunk_transcript(self, transcript_path, chunk_size=964):
        with open(transcript_path, 'r') as f:
            transcript = json.load(f)
            speaker_frames = transcript['transcript']['content'][0]['content']

            utterances = []
            for x in range(len(speaker_frames)):
                speaker_name = transcript['speaker_info'][speaker_frames[x]['attrs']['speakerId']-1]['name']
                utterances.append(f'{speaker_name}: ' + ''.join([i['content'][0]['text'] for i in speaker_frames[x]['content']]))

        chunked = [''.join(utterances)[i: i + chunk_size] for i in range(0, len(''.join(utterances)), chunk_size)]
        return chunked
    

    def extract_keyphrases(self, batches):
        summaries = []
        for utterances in batches:
            inputs = self.tokenizer([utterances], max_length=1024, return_tensors="pt")
            summary_ids = self.model.generate(inputs["input_ids"], num_beams=4, min_length=0, max_length=200)
            decoded = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            summaries.append(decoded)
    
        return summaries