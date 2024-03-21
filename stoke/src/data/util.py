import re
from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok
from torch.utils.data import Dataset
import json
import torch
import torch.nn.functional as F
import os
import datasets
import random



class Detokenizer(object):
    # https://stackoverflow.com/a/46311499
    def __init__(self) -> None:
        self.detokenizer = Detok()

    def __call__(self, tokens, return_offsets=False):
        text = self.detokenizer.detokenize(tokens)
        text = re.sub('\s*,\s*', ', ', text)
        text = re.sub('\s*\.\s*', '. ', text)
        text = re.sub('\s*\?\s*', '? ', text)
        text = text.replace(" --", "--")

        if return_offsets:
            offsets = [0]
            for i in range(1, len(tokens)):
                offsets.append(len(self(tokens[:i])))

            """
            # verify offsets
            for i, offset in enumerate(offsets):
                if i == 0:
                    continue
                check = text[:offset]
                target = self(tokens[:i])
                try:
                    assert target == check
                except AssertionError:
                    print(tokens)
                    print(f"'{check}' != '{target}'")
                    raise AssertionError
            """

            return text.strip(), offsets
        return text.strip()

class JSONDataset(Dataset):

    def __init__(self, path):
        super().__init__()
        
        self.samples = json.load(open(path, "r"))

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]


def create_mask_for_len(seq_len, pad_to=None, skip_start=0, window=None):
    mask = (-1 * (torch.triu(torch.ones(seq_len, seq_len), diagonal=1) - 1)).bool()

    if skip_start != 0:
        mask[:skip_start, :] = False
        mask[:, :skip_start] = False
    
    if window is not None:
        for i in range(window, seq_len):
            mask[i, :max(i-window, 0)] = False

    if pad_to is None:
        return mask
    
    return F.pad(mask, (0, pad_to-seq_len, 0, pad_to-seq_len))


def collate_function_with_label_map(batch, label_map):
    # prepare token ids
    sequences = [torch.tensor(x['tokens']) for x in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    
    # prepare labels
    labels_tokens = []
    for batchitem in batch:
        labels_tokens.append(torch.tensor([label_map.index(x) for x in batchitem['ner_tags']]))
    labels_tokens = torch.nn.utils.rnn.pad_sequence(labels_tokens, batch_first=True, padding_value=0).to(torch.long)

    labels_spans = torch.zeros((len(batch), input_ids.shape[-1], input_ids.shape[-1]))
    for i, batchitem in enumerate(batch):
        for mnt in batchitem['mentions']:
            start, end = mnt
            try:
                labels_spans[i, end+1, start] = 1.0
            except:
                pass

    # prepare masks
    masks_tokens = [torch.tensor([False]+([True]*(len(x)-1))) for x in sequences]
    masks_tokens = torch.nn.utils.rnn.pad_sequence(masks_tokens, batch_first=True, padding_value=False)
    mask_spans = torch.stack([create_mask_for_len(len(x['tokens']), input_ids.shape[-1], skip_start=0, window=15) for x in batch])

    # mask labels
    labels_tokens = torch.masked_select(labels_tokens, masks_tokens).long()
    labels_spans = torch.masked_select(labels_spans, mask_spans).long()

    return {
        'input_ids': input_ids,
        'labels_tokens': labels_tokens,
        'labels_spans': labels_spans,
        'mask_tokens': masks_tokens.unsqueeze(-1),
        'mask_spans': mask_spans.unsqueeze(-1)
        }

def print_metric(metric, class_labels, return_classwise=False, verbose=False):
    
    f_ner = metric.compute()
    p_ner = torch.nan_to_num(metric.num_tp / metric.num_prediction)
    r_ner = torch.nan_to_num(metric.num_tp / metric.num_label)

    if verbose:
        print(f"{' '.ljust(10)}     P      R      F      S")

    sum_support = 0
    weighted_scores = [0, 0, 0]

    classwise = {}
    for ner_class, p, r, f, s in zip(class_labels, p_ner, r_ner, f_ner, metric.num_label):
        if ner_class == "NONE" or ner_class == "O" or ner_class == "no_relation" or ner_class == "no_span":
            continue
        if verbose:
            print(f"{ner_class.ljust(10)} - {p:.2f} - {r:.2f} - {f:.2f} - {int(s)}")
        weighted_scores[0] += p*s
        weighted_scores[1] += r*s
        weighted_scores[2] += f*s
        sum_support += s

        classwise[ner_class] = {"p": p.item(), "r": r.item(), "f": f.item(), "s": s.item()}

    p_micro = weighted_scores[0]/sum_support
    r_micro = weighted_scores[1]/sum_support
    f_micro = weighted_scores[2]/sum_support

    classwise["macro"] = {"p": torch.mean(p_ner[1:]).item(), "r": torch.mean(r_ner[1:]).item(), "f": torch.mean(f_ner[1:]).item()}

    if verbose:
        print("")
        print(f"MICRO      - {p_micro:.2f} - {r_micro:.2f} - {f_micro:.2f}")
        print(f"MACRO      - {torch.mean(p_ner[1:]):.2f} - {torch.mean(r_ner[1:]):.2f} - {torch.mean(f_ner[1:]):.2f}")
        print("")

    if return_classwise:
        return (p_micro.item(), r_micro.item(), f_micro.item()), classwise

    return p_micro.item(), r_micro.item(), f_micro.item()

class GenerationConfig:
    
    def __init__(self, language_model, output_path, dataset_name, cuda=False, generation_kwargs={}):
        self.language_model = language_model
        self.output_path = output_path
        self.dataset_name = dataset_name
        self.cuda = cuda
        self.generation_kwargs = generation_kwargs
        
        self.path_data = os.path.join(output_path, f"{language_model}/{dataset_name}/data.json")
        self.path_config = os.path.join(output_path, f"{language_model}/{dataset_name}/config.json")
        
        if not os.path.exists(os.path.join(output_path, f"{language_model}/{dataset_name}")):
            os.makedirs(os.path.join(output_path, f"{language_model}/{dataset_name}"))
        
        
def conll_prompts():
    ds = datasets.load_dataset("conll2003")["validation"]
    dtk = Detokenizer()
    prompts = [dtk(x["tokens"]) for x in ds]
    ds = datasets.load_dataset("conll2003")["train"]
    prompts += [dtk(x["tokens"]) for x in ds]
    return prompts


def partition_dataset(data, split_sizes):
    random.shuffle(data)
    
    total_size = len(data)
    split_points = [int(total_size * size) for size in split_sizes[:-1]]
    
    datasets = []
    start_idx = 0
    for split_point in split_points:
        datasets.append(data[start_idx: start_idx + split_point])
        start_idx += split_point
    datasets.append(data[start_idx:])
    
    return datasets

def stats(ds, keys=None):
    mentions_total = 0
    mentions_per_type = {}
    if keys is not None:
        for key in keys:
            mentions_per_type[key] = 0
    for sample in ds:
        mentions_total += len(sample['mentions'])
        for mnt in sample['mentions']:
            tag = sample['ner_tags'][mnt[0]]
            if tag not in mentions_per_type.keys():
                mentions_per_type[tag] = 0
            mentions_per_type[tag] += 1
    return len(mentions_per_type.keys()), mentions_total, mentions_per_type


def split_data(path_to_data, split_names=["train", "validation", "test"], split_sizes=[0.8, 0.1, 0.1]):
    with open(path_to_data, 'r') as file:
        data = json.load(file)
    
    annotation_types = sorted(list(stats(data)[-1].keys()))
    datasets = partition_dataset(data, split_sizes)

    for i, dataset in enumerate(datasets):
        ds = []
        for x in dataset:
            out = {}
            out["tokens"] = x["tokens"] 
            out["ner_tags"] = [y.replace("I-", "").replace("B-", "") for y in x["ner_tags"]]
            out["mentions"] = x["mentions"] 
            ds.append(out)

        print(f"Size of dataset {split_names[i]}: {len(ds)}")
        json.dump(ds, open(f"{path_to_data.split('.json')[0]}_{split_names[i]}.json", "w"))
        json.dump(stats(dataset, annotation_types)[-1], open(f"{path_to_data.split('.json')[0]}_{split_names[i]}_stats.json", "w"), indent=1)
    