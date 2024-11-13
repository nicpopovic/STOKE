import argparse
from stoke.src.trainer.util import TrainConfig
from stoke.src.trainer.trainer import Trainer
from stoke.src.selection.simple import create_config_for_path

def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument('--path', type=str, default='data/meta-llama/Llama-3.2-1B-Instruct/STOKE_500_wikiqa', help='Path to the training data')
    parser.add_argument('--layers', type=int, nargs='+', default=[8, 9, 10, 11, 12], help='List of layers')
    parser.add_argument('--learning_rates', type=float, nargs='+', default=[1e-4, 5e-5, 3e-4, 5e-4, 1e-3], help='List of learning rates')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--n_steps_per_epoch', type=int, default=500, help='Number of steps per epoch')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    return parser.parse_args()

def main():
    args = parse_args()

    # create TrainConfig object with values from argparse
    config = TrainConfig(
        args.path,
        layers=args.layers,
        learning_rates=args.learning_rates,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        n_steps_per_epoch=args.n_steps_per_epoch,
        cuda=args.cuda
    )

    # create Trainer
    trainer = Trainer(config)

    # run training
    trainer.train()

    # create basic config for playground
    create_config_for_path(config.path, "basic")

if __name__ == "__main__":
    main()
