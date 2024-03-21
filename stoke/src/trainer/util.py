import os
import time
import json


class TrainConfig:
    
    def __init__(self, path, splits=['train', 'validation'],
                 layers=[9, 10, 11], hfcache='', classifier_dims=[4096], learning_rates=[1e-4],
                 cuda=False, n_steps_per_epoch=1000, n_epochs=2, batch_size=8, balance_loss=False,
                 loss_weights_span=[[1.0, 1.0], [1.0, 50.0], [1.0, 100.0]]):
        self.path = path
        self.checkpoint_path = os.path.join(self.path, "checkpoints/")
        self.splits = splits
        self.layers = layers
        self.hfcache = hfcache
        self.classifier_dims = classifier_dims
        self.learning_rates = learning_rates
        self.cuda = cuda
        self.n_steps_per_epoch = n_steps_per_epoch
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.balance_loss = balance_loss
        self.loss_weights_span = loss_weights_span
        self.time = time.time()
        self.config_dataset = json.load(open(os.path.join(path, f"config.json"), "r"))

    def to_dict(self):
        return {
            "path": self.path,
            "splits": self.splits,
            "layers": self.layers,
            "hfcache": self.hfcache,
            "classifier_dims": self.classifier_dims,
            "learning_rates": self.learning_rates,
            "cuda": self.cuda,
            "n_steps_per_epoch": self.n_steps_per_epoch,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "balance_loss": self.balance_loss,
            "loss_weights_span": self.loss_weights_span,
            "time": self.time,
            "config_dataset": self.config_dataset
        }