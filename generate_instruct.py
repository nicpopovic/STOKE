import argparse
from stoke.src.data.util import GenerationConfig, split_data, conll_prompts, wikiqa_prompts
from stoke.src.data.generation import DataGenerator, FlairNERModel
import random

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run data generation and annotation pipeline.")
    
    # Add arguments
    parser.add_argument('--language_model', type=str, default="meta-llama/Llama-3.2-1B-Instruct", 
                        help='Language model to be used for generation.')
    parser.add_argument('--output_path', type=str, default="data/", 
                        help='Path to save the generated data.')
    parser.add_argument('--dataset_name', type=str, default="STOKE_500_wikiqa", 
                        help='Name of the dataset to be used.')
    parser.add_argument('--cuda', action='store_true', 
                        help='Flag to indicate if CUDA should be used for generation.')
    parser.add_argument('--max_new_tokens', type=int, default=500, 
                        help='Maximum number of new tokens to be generated.')
    parser.add_argument('--repetition_penalty', type=float, default=1.2, 
                        help='Repetition penalty to be applied during generation.')

    # Parse arguments
    args = parser.parse_args()

    # generation parameters
    generation_kwargs = {"max_new_tokens": args.max_new_tokens, "repetition_penalty": args.repetition_penalty}

    # Creating TrainConfig object with values from argparse
    config = GenerationConfig(language_model=args.language_model, 
                              output_path=args.output_path, 
                              dataset_name=args.dataset_name, 
                              cuda=args.cuda, 
                              generation_kwargs=generation_kwargs)

    # create annotation model
    reference_model = FlairNERModel(config.language_model, "flair/ner-english-ontonotes-large")

    # create DataGenerator
    generator = DataGenerator(config, reference_model)

    # Prepare prompts for generation
    prompts = [generator.tokenizer.apply_chat_template([{"role": "user", "content": x}], 
                                                       tokenize=False, add_generation_prompt=True)
               .replace("<|begin_of_text|>", "") for x in wikiqa_prompts()]

    # run generator
    generated_texts = generator.generate_text(prompts, generation_kwargs)

    # annotate text with reference model
    annotated_texts = generator.annotate_text(generated_texts)

    # save data in correct format
    generator.save_data(annotated_texts)

    # split dataset
    split_data(config.path_data)

if __name__ == "__main__":
    main()
