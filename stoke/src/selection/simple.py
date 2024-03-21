import json
import os


def find_best_checkpoint(path):
    checkpoints_path = os.path.join(path, "checkpoints")
    token_classifier_path = os.path.join(checkpoints_path, "token_classifier")
    span_classifier_path = os.path.join(checkpoints_path, "span_classifier")

    best_token_checkpoint = find_best_checkpoint_in_folder(token_classifier_path)
    best_span_checkpoint = find_best_checkpoint_in_folder(span_classifier_path)

    return best_token_checkpoint, best_span_checkpoint

def find_best_checkpoint_in_folder(folder_path):
    best_checkpoint = None
    best_f1_validation = -1

    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        config_path = os.path.join(subfolder_path, "config.json")
        checkpoint_path = os.path.join(subfolder_path, "checkpoint.pt")

        if os.path.exists(config_path) and os.path.exists(checkpoint_path):
            with open(config_path, 'r') as config_file:
                config_data = json.load(config_file)
                if "best_f1_validation" in config_data:
                    f1_validation = config_data["best_f1_validation"]
                    if f1_validation > best_f1_validation:
                        best_f1_validation = f1_validation
                        best_checkpoint = subfolder_path

    return best_checkpoint

def create_config_for_path(path, name="default"):
    
    best_token_checkpoint, best_span_checkpoint = find_best_checkpoint(path)

    print("Best token classifier checkpoint:", best_token_checkpoint)
    print("Best span classifier checkpoint:", best_span_checkpoint)

    config = {
        "classifier_token": best_token_checkpoint,
        "classifier_span": best_span_checkpoint
    }

    configs_path = os.path.join(path, "stoke_config.json")

    if os.path.exists(configs_path):
        with open(configs_path, 'r') as configs_file:
            existing_configs = json.load(configs_file)
    else:
        existing_configs = {}

    existing_configs[name] = config

    with open(configs_path, 'w') as configs_file:
        json.dump(existing_configs, configs_file, indent=4)

    print(f"Config '{name}' saved successfully.")
