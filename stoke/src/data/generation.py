from transformers import pipeline
from tqdm import tqdm
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from flair.models import SequenceTagger
from flair.data import Sentence
import os



class AnnotationModel:
    def __init__(self, model_id_for_tokenizer):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_for_tokenizer, use_fast=True, token=os.getenv("HF_TOKEN"))
        self.pipe = pipeline("token-classification", model="FacebookAI/xlm-roberta-large-finetuned-conll03-english", aggregation_strategy="simple")

    def annotate_text(self, text):
        iob_tags = ['O'] * len(text)
        mentions = []
        text_str = self.tokenizer.decode(text)
        ner_tags = self.pipe(text_str)
        
        offsets = []
        offset = 0
        for i, token_id in enumerate(text):
            offsets.append(offset)
            offset = len(self.tokenizer.decode(text[:i+1]))
        offsets.append(offset)
        
        for tag in ner_tags:
            try:
                start = self.get_token_for_char(tag["start"], offsets)
                end = self.get_token_for_char(tag["end"]-1, offsets)
                mentions.append([start, end])
                for i in range(start, end+1):
                    #iob_tags[i] = "I-" + tag["entity_group"]
                    iob_tags[i] = tag["entity_group"]
                #iob_tags[start] = "B-" + tag["entity_group"]
                iob_tags[start] = tag["entity_group"]
            except Exception as e:
                print(e)
                pass
        
        return {"tokens": text, "ner_tags": iob_tags, "mentions": mentions}

    def get_token_for_char(self, i, offsets):
        for off in range(len(offsets)):
            if i < offsets[off]:
                return off - 1
        return len(offsets) - 1
    
class FlairNERModel:
    def __init__(self, model_id_for_tokenizer, flair_model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_for_tokenizer, use_fast=True, token=os.getenv("HF_TOKEN"))
        self.tagger = SequenceTagger.load(flair_model_name)
        self.name = flair_model_name

    def annotate_text(self, text):
        iob_tags = ['O'] * len(text)
        mentions = []
        text_str = self.tokenizer.decode(text)
        sentence = Sentence(text_str)
        
        # Predict NER tags
        self.tagger.predict(sentence)
        
        ner_tags = sentence.get_spans('ner')
        
        offsets = []
        offset = 0
        for i, token_id in enumerate(text):
            offsets.append(offset)
            offset = len(self.tokenizer.decode(text[:i+1]))
        offsets.append(offset)
        
        for tag in ner_tags:
            try:
                start = self.get_token_for_char(tag.start_position, offsets)
                end = self.get_token_for_char(tag.end_position-1, offsets)
                mentions.append([start, end])
                for i in range(start, end+1):
                    #iob_tags[i] = "I-"+ tag.get_labels('ner')[0].to_dict()['value']
                    iob_tags[i] = tag.get_labels('ner')[0].to_dict()['value']
                #iob_tags[start] = "B-"+ tag.get_labels('ner')[0].to_dict()['value']
                iob_tags[start] = tag.get_labels('ner')[0].to_dict()['value']
                #print(tag, self.tokenizer.decode(text[start:end+1]))
            except Exception as e:
                print(tag)
                print(e)
                pass
        
        return {"tokens": text, "ner_tags": iob_tags, "mentions": mentions}

    def get_token_for_char(self, i, offsets):
        for off in range(len(offsets)):
            if i < offsets[off]:
                return off - 1
        return len(offsets) - 1    
    
    
class FlairChunkingModel:
    def __init__(self, model_id_for_tokenizer, flair_model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_for_tokenizer, use_fast=True, token=os.getenv("HF_TOKEN"))
        self.tagger = SequenceTagger.load(flair_model_name)

    def annotate_text(self, text):
        iob_tags = ['O'] * len(text)
        mentions = []
        text_str = self.tokenizer.decode(text)
        sentence = Sentence(text_str)
        
        # Predict NER tags
        self.tagger.predict(sentence)
        
        ner_tags = sentence.get_spans('np')
        
        offsets = []
        offset = 0
        for i, token_id in enumerate(text):
            offsets.append(offset)
            offset = len(self.tokenizer.decode(text[:i+1]))
        offsets.append(offset)
        
        for tag in ner_tags:
            try:
                start = self.get_token_for_char(tag.start_position, offsets)
                end = self.get_token_for_char(tag.end_position-1, offsets)
                mentions.append([start, end])
                for i in range(start, end+1):
                    #iob_tags[i] = "I-"+ tag.get_labels('ner')[0].to_dict()['value']
                    iob_tags[i] = tag.get_labels('np')[0].to_dict()['value']
                #iob_tags[start] = "B-"+ tag.get_labels('ner')[0].to_dict()['value']
                iob_tags[start] = tag.get_labels('np')[0].to_dict()['value']
                #print(tag, self.tokenizer.decode(text[start:end+1]))
            except Exception as e:
                print(tag)
                print(e)
                pass
        
        return {"tokens": text, "ner_tags": iob_tags, "mentions": mentions}

    def get_token_for_char(self, i, offsets):
        for off in range(len(offsets)):
            if i < offsets[off]:
                return off - 1
        return len(offsets) - 1
    
    
class FlairFrameModel:
    def __init__(self, model_id_for_tokenizer, flair_model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_for_tokenizer, use_fast=True, token=os.getenv("HF_TOKEN"))
        self.tagger = SequenceTagger.load(flair_model_name)

    def annotate_text(self, text):
        iob_tags = ['O'] * len(text)
        mentions = []
        text_str = self.tokenizer.decode(text)
        sentence = Sentence(text_str)
        
        # Predict NER tags
        self.tagger.predict(sentence)

        ner_tags = sentence.get_labels('frame')
        
        
        offsets = []
        offset = 0
        for i, token_id in enumerate(text):
            offsets.append(offset)
            offset = len(self.tokenizer.decode(text[:i+1]))
        offsets.append(offset)
        
        for tag in ner_tags:
            try:
                start = self.get_token_for_char(tag.data_point.start_position, offsets)
                end = self.get_token_for_char(tag.data_point.end_position-1, offsets)
                mentions.append([start, end])
                for i in range(start, end+1):
                    #iob_tags[i] = "I-"+ tag.get_labels('ner')[0].to_dict()['value']
                    iob_tags[i] = tag.to_dict()['value']
                #iob_tags[start] = "B-"+ tag.get_labels('ner')[0].to_dict()['value']
                iob_tags[start] = tag.to_dict()['value']
                #print(tag, self.tokenizer.decode(text[start:end+1]))
            except Exception as e:
                print(tag)
                print(e)
                pass
        
        return {"tokens": text, "ner_tags": iob_tags, "mentions": mentions}

    def get_token_for_char(self, i, offsets):
        for off in range(len(offsets)):
            if i < offsets[off]:
                return off - 1
        return len(offsets) - 1
    
    
class FlairPOSModel:
    def __init__(self, model_id_for_tokenizer, flair_model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id_for_tokenizer, use_fast=True, token=os.getenv("HF_TOKEN"))
        self.tagger = SequenceTagger.load(flair_model_name)

    def annotate_text(self, text):
        iob_tags = ['O'] * len(text)
        mentions = []
        text_str = self.tokenizer.decode(text)
        sentence = Sentence(text_str)
        
        # Predict NER tags
        self.tagger.predict(sentence)

        ner_tags = sentence.get_labels('pos')
        
        
        offsets = []
        offset = 0
        for i, token_id in enumerate(text):
            offsets.append(offset)
            offset = len(self.tokenizer.decode(text[:i+1]))
        offsets.append(offset)
        
        for tag in ner_tags:
            try:
                start = self.get_token_for_char(tag.data_point.start_position, offsets)
                end = self.get_token_for_char(tag.data_point.end_position-1, offsets)
                mentions.append([start, end])
                for i in range(start, end+1):
                    #iob_tags[i] = "I-"+ tag.get_labels('ner')[0].to_dict()['value']
                    iob_tags[i] = tag.to_dict()['value']
                #iob_tags[start] = "B-"+ tag.get_labels('ner')[0].to_dict()['value']
                iob_tags[start] = tag.to_dict()['value']
                #print(tag, self.tokenizer.decode(text[start:end+1]))
            except Exception as e:
                print(tag)
                print(e)
                pass
        
        return {"tokens": text, "ner_tags": iob_tags, "mentions": mentions}

    def get_token_for_char(self, i, offsets):
        for off in range(len(offsets)):
            if i < offsets[off]:
                return off - 1
        return len(offsets) - 1

class DataGenerator(object):
    def __init__(self, config, reference_model):
        self.config = config
        self.model_id = self.config.language_model
        self.reference_model = reference_model
        self.output_path = self.config.path_data
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.language_model, use_fast=True, token=os.getenv("HF_TOKEN"))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        if self.config.cuda:
            device = "cuda"
            self.model = AutoModelForCausalLM.from_pretrained(self.config.language_model, token=os.getenv("HF_TOKEN"), device_map="auto")
        else:
            device = "cpu"    
            self.model = AutoModelForCausalLM.from_pretrained(self.config.language_model, token=os.getenv("HF_TOKEN")).to(device)
        
        json.dump({
            "generation_kwargs": self.config.generation_kwargs,
            "model_id": self.config.language_model,
            "flair_model_name": reference_model.name,
        }, open(self.config.path_config, "w"), indent=1)
    

    def generate_text(self, prompts, generation_kwargs):
        generated_texts = []
        for prompt in tqdm(prompts, desc="Generating text"):
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            generated_text_ids = self.model.generate(input_ids=input_ids.to(self.model.device), pad_token_id=self.tokenizer.pad_token_id, **generation_kwargs)
            generated_text = generated_text_ids[0].tolist()
            prompt_token_ids = input_ids[0].tolist()
            generated_texts.append({"prompt": prompt_token_ids, "full": generated_text})
        return generated_texts

    def annotate_text(self, texts):
        annotated_texts = []
        for text in tqdm(texts, desc="Annotating text"):
            annotated_text = self.reference_model.annotate_text(text["full"])
            annotated_texts.append(annotated_text)
        return annotated_texts

    def save_data(self, data):
        with open(self.output_path, 'w') as f:
            json.dump(data, f, indent=1)

