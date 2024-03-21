from .util import TrainConfig
from ..data.util import JSONDataset, collate_function_with_label_map, print_metric
from ..classifier.probes import MLPProbe as MLP
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassF1Score
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
import json
import os
import string
import random
from tqdm import tqdm


class Trainer:
    
    def __init__(self, config:TrainConfig):
        self.config = config
        self._load_model()
        self._load_data()
        self._load_probes_and_optimizers()
        print("Trainer is ready.")
    
    def _load_model(self):
        "Loads language model and tokenizer"
        print(f"Loading model '{self.config.config_dataset['model_id']}'")
        
        # check if custom huggingface cache was selected
        kwds = {
        }
        if self.config.hfcache != "":
            kwds["cache_dir"] = self.config.hfcache

        # load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.config.config_dataset['model_id'], output_attentions=True, output_hidden_states=True, return_dict=True, device_map="auto", **kwds).half()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.config_dataset['model_id'], use_fast=True, **kwds)
        
        if self.config.cuda:
            self.model.cuda()
        
        print("model and tokenizer loaded")
        
    def _load_data(self):
        "Loads datasets"
        def collate_function(batch):
            return collate_function_with_label_map(batch, self.config.label_map)

        datasets = {}
        self.dataloaders = {}
        num_classes = None
        self.config.label_map = []
        for split in self.config.splits:
            datasets[split] = JSONDataset(os.path.join(self.config.path, f"data_{split}.json"))
            shuffle = False
            if split == "train":
                shuffle = True
            self.dataloaders[split] = DataLoader(datasets[split], batch_size=self.config.batch_size, shuffle=shuffle, collate_fn=collate_function)

            dataset_classes = json.load(open(os.path.join(self.config.path, f"data_{split}_stats.json"), "r"))
            if num_classes is None:
                num_classes = len(dataset_classes.keys())
                self.config.label_map = ["O"] + list(dataset_classes.keys())
            else:
                assert len(dataset_classes.keys()) == num_classes
            print(f"Loaded {split} dataset with {len(datasets[split])} samples and {num_classes} classes")

    def _load_probes_and_optimizers(self):
        "Loads probes and optimizers."
        print("Preparing probes and optimizers")
        if type(self.model.config):
            n_layers = self.model.config.num_hidden_layers
            n_heads = self.model.config.num_attention_heads
            dim_hidden = self.model.config.hidden_size

        print(f"Model has {n_layers} layers, hidden state size {dim_hidden}, and {n_heads} attention heads per layer")

        if self.config.layers is None:
            self.config.layers = [x for x in range(n_layers)]

        print(f"Training tokenwise classifiers for {len(self.config.layers)} layer(s), {len(self.config.learning_rates)} learning rate(s), {len(self.config.classifier_dims)} hidden dims.")

        self.classifier_device = "cpu"
        if self.config.cuda:
            self.classifier_device = "cuda"

        if self.config.balance_loss:
            class_frequency = [0 for _ in self.config.label_map]
            for sample in self.dataloaders["train"]:
                labels = [self.config.label_map.index(x) for x in sample['ner_tags']]
                for x in labels:
                    class_frequency[x] += 1

            class_weights = [sum(class_frequency)/x for x in class_frequency]
        else:
            class_weights = [1.0 for _ in self.config.label_map]


        self.token_classifiers = []
        for layer in self.config.layers:
            for lr in self.config.learning_rates:
                for dim_c in self.config.classifier_dims:
                    # set up classifier, optimizer, scheduler, and config
                    _classifier = MLP(dim_hidden, len(self.config.label_map), hidden_dim=dim_c, cuda=self.config.cuda)
                    _optimizer = AdamW(_classifier.parameters(), lr=lr, eps=1e-6)
                    _scheduler = get_linear_schedule_with_warmup(_optimizer, self.config.n_steps_per_epoch, self.config.n_epochs*self.config.n_steps_per_epoch)
                    _config = {
                            "layer": layer,
                            "model": self.config.config_dataset['model_id'],
                            "type": "token_classifier",
                            "label_map": self.config.label_map,
                            "learning_rate": lr,
                            "classifier_dim": dim_c,
                            "loss_weights": class_weights,
                            "identifier": ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(10)),
                            "best_f1_validation": -1,
                            "best_f1_validation_classwise": 0,
                        }
                    self.token_classifiers.append({
                        "config_train": self.config,
                        "config": _config,
                        "classifier": _classifier,
                        "optimizer": _optimizer,
                        "lr_scheduler": _scheduler,
                        "metric": MulticlassF1Score(num_classes=len(self.config.label_map), average=None, device=_classifier.device),
                        "criterion": torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(self.classifier_device))


                    })

        print(f"Total tokenwise classifiers: {len(self.token_classifiers)}")

        print(f"Training span detectors for {len(self.config.learning_rates)} learning rate(s), {len(self.config.loss_weights_span)} loss weight(s), {len(self.config.classifier_dims)} hidden dim(s).")

        self.span_classifiers = []
        for lr in self.config.learning_rates:
            for dim_c in self.config.classifier_dims:
                for loss_weight in self.config.loss_weights_span:
                    # set up classifier, optimizer, scheduler, and config
                    _classifier = MLP(n_layers*n_heads, 2, hidden_dim=dim_c, cuda=self.config.cuda)
                    _optimizer = AdamW(_classifier.parameters(), lr=lr, eps=1e-6)
                    _scheduler = get_linear_schedule_with_warmup(_optimizer, self.config.n_steps_per_epoch, self.config.n_epochs*self.config.n_steps_per_epoch)
                    _config = {
                            "model": self.config.config_dataset['model_id'],
                            "type": "span_classifier",
                            "label_map": ["no_span", "span"],
                            "learning_rate": lr,
                            "classifier_dim": dim_c,
                            "loss_weights": loss_weight,
                            "identifier": ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(10)),
                            "best_f1_validation": -1,
                            "best_f1_validation_classwise": 0,
                        }
                    self.span_classifiers.append({
                        "config_train": self.config,
                        "config": _config,
                        "classifier": _classifier,
                        "optimizer": _optimizer,
                        "lr_scheduler": _scheduler,
                        "metric": MulticlassF1Score(num_classes=2, average=None, device=_classifier.device),
                        "criterion": torch.nn.CrossEntropyLoss(weight=torch.tensor(loss_weight).to(self.classifier_device))
                    })
        print(f"Total span detectors: {len(self.span_classifiers)}")

    def train(self):
        data_iter_train = iter(self.dataloaders["train"])

        self.best_f1 = {"token_classifier":-1, "span_classifier":-1}
        self.best_config = {"token_classifier":None, "span_classifier":None}


        for epoch in range(self.config.n_epochs):

            # TRAIN
            for item in self.token_classifiers + self.span_classifiers:
                item['classifier'].train()
                item['metric'].reset()

            for step in tqdm(range(self.config.n_steps_per_epoch)):

                # Get data
                try:
                    sample = next(data_iter_train)
                except StopIteration:
                    data_iter_train = iter(self.dataloaders["train"])
                    sample = next(data_iter_train)

                with torch.no_grad():
                    input_ids = sample['input_ids']
                    labels_tokens = sample['labels_tokens']
                    labels_spans = sample['labels_spans']

                    outputs = self.model(input_ids.to(self.model.device), output_hidden_states=True, output_attentions=True)

                    hidden_states = {}
                    for layer in self.config.layers:
                        hidden_states[layer] = outputs.hidden_states[layer].to(self.classifier_device)

                    # get attentions and labels
                    attentions = torch.stack(outputs.attentions).swapaxes(0,1)
                    attentions = attentions.reshape(attentions.size(0), -1, attentions.size(-2), attentions.size(-1)).permute(0, 2, 3, 1)
                    attentions = torch.masked_select(attentions, sample['mask_spans'].to(self.classifier_device)).view(-1, attentions.size(-1))


                # training step for each classifier
                for item in self.span_classifiers + self.token_classifiers:
                    if item['config']['type'] == "span_classifier":
                        _preds = item['classifier'](attentions.to(item['classifier'].fc1.weight.dtype).to(self.classifier_device))
                        _labels = labels_spans.to(self.classifier_device)
                    elif item['config']['type'] == "token_classifier":
                        _preds = item['classifier'](hidden_states[item['config']['layer']].to(item['classifier'].fc1.weight.dtype))
                        _preds = torch.masked_select(_preds, sample['mask_tokens'].to(self.classifier_device))
                        _labels = labels_tokens.to(self.classifier_device)

                    loss = item['criterion'](_preds.view(-1, len(item['config']['label_map'])), _labels.view(-1))
                    item['metric'].update(_preds.view(-1, len(item['config']['label_map'])), _labels.view(-1))

                    item['optimizer'].zero_grad(set_to_none=True)
                    loss.backward()
                    item['optimizer'].step()
                    item['lr_scheduler'].step()

                hidden_states = {}
                attentions = None

            # EVAL
            for item in self.span_classifiers + self.token_classifiers:
                item['classifier'].eval()
                item['metric'].reset()

            with torch.no_grad():
                
                for sample in tqdm(self.dataloaders["validation"]):
                    input_ids = sample['input_ids']
                    labels_tokens = sample['labels_tokens']
                    labels_spans = sample['labels_spans']

                    # language model forward pass
                    outputs = self.model(input_ids.to(self.model.device), output_hidden_states=True, output_attentions=True)

                    # get internal representations into correct shapes
                    for layer in self.config.layers:
                        hidden_states[layer] = outputs.hidden_states[layer].to(self.classifier_device)
                    attentions = torch.stack(outputs.attentions).swapaxes(0,1)
                    attentions = attentions.reshape(attentions.size(0), -1, attentions.size(-2), attentions.size(-1)).permute(0, 2, 3, 1)
                    attentions = torch.masked_select(attentions, sample['mask_spans'].to(self.classifier_device)).view(-1, attentions.size(-1))

                    # classifier inference
                    for item in self.span_classifiers + self.token_classifiers:
                        if item['config']['type'] == "span_classifier":
                            _preds = item['classifier'](attentions.to(item['classifier'].fc1.weight.dtype).to(self.classifier_device))
                            _labels = labels_spans.to(self.classifier_device)
                        elif item['config']['type'] == "token_classifier":
                            _preds = item['classifier'](hidden_states[item['config']['layer']].to(item['classifier'].fc1.weight.dtype))
                            _preds = torch.masked_select(_preds, sample['mask_tokens'].to(self.classifier_device))
                            _labels = labels_tokens.to(self.classifier_device)
                        item['metric'].update(_preds.view(-1, len(item['config']['label_map'])), _labels.view(-1))

                # logging and saving of checkpoints
                for item in self.span_classifiers + self.token_classifiers:
                    (p_micro, r_micro, f_micro), classwise = print_metric(item['metric'], item['config']['label_map'], return_classwise=True, verbose=False)
                    if f_micro > item['config']['best_f1_validation']:
                        item['config']['best_f1_validation'] = f_micro
                        item['config']['best_f1_validation_classwise'] = classwise

                        ckp_path = os.path.join(self.config.checkpoint_path, f"{item['config']['type']}/{item['config']['identifier']}/")
                        os.makedirs(ckp_path, exist_ok=True)
                        torch.save(item['classifier'].state_dict(), os.path.join(ckp_path, f"checkpoint.pt"))
                        json.dump(item['config'], open(os.path.join(ckp_path, f"config.json"), "w"), indent=1)
                        json.dump(item['config_train'].to_dict(), open(os.path.join(ckp_path, f"config_train.json"), "w"), indent=1)

                    if f_micro > self.best_f1[item['config']['type']]:
                        self.best_f1[item['config']['type']] = f_micro
                        self.best_config[item['config']['type']] = item['config']
            
            # print current best for each classifier type
            for key in self.best_config.keys():
                print(f"--- Best {key} config after epoch {epoch+1} ---")
                if self.best_config[key] is not None:
                    for key, value in self.best_config[key].items():
                        print(key, value)
        return self.best_config

