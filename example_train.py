# imports
from stoke.src.trainer.util import TrainConfig
from stoke.src.trainer.trainer import Trainer
from stoke.src.selection.simple import create_config_for_path

# create TrainConfig object with default values
config = TrainConfig('data/gpt2/test', n_steps_per_epoch=10, n_epochs=10)

# create Trainer
trainer = Trainer(config)

# run training
trainer.train()

# create basic config for playground
create_config_for_path(config.path, "basic")
