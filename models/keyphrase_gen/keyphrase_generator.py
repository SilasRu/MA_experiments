from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json



class KeyphraseGenerator:
    def __init__(self, model_path) -> None:
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def _chunk_text(self, text, chunk_size=1024):
        chunked = [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]
        return chunked

    def preprocess_and_chunk_transcript(self, transcript_path, chunk_size=1024):
        with open(transcript_path, 'r') as f:
            transcript = json.load(f)
            speaker_frames = transcript['transcript']['content'][0]['content']

            utterances = []
            for x in range(len(speaker_frames)):
                speaker_name = transcript['speaker_info'][speaker_frames[x]['attrs']['speakerId']-1]['name']
                utterances.append(f'{speaker_name}: ' + ''.join([i['content'][0]['text'] for i in speaker_frames[x]['content']]))
            utterances = ''.join(utterances)
        chunked = self._chunk_text(utterances, chunk_size)
        return chunked
    

    def extract_keyphrases(self, batches):
        summaries = []
        for utterances in batches:
            inputs = self.tokenizer([utterances], max_length=1024, return_tensors="pt", truncation=True)
            summary_ids = self.model.generate(inputs["input_ids"], num_beams=4, min_length=0, max_length=200)
            decoded = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            summaries.append(decoded)
    
        return summaries