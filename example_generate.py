from stoke.src.data.util import GenerationConfig, split_data, conll_prompts
from stoke.src.data.generation import DataGenerator, FlairNERModel

# generation parameters
generation_kwargs = {"max_new_tokens": 100, "repetition_penalty": 1.2}

# Creating TrainConfig object with default values
config = GenerationConfig(language_model="gpt2", output_path="data/", dataset_name="test", cuda=False, generation_kwargs=generation_kwargs)

# create annotation model
reference_model = FlairNERModel(config.language_model, "flair/ner-english-ontonotes-large")

# create DataGenerator
generator = DataGenerator(config, reference_model)

# run generator
generated_texts = generator.generate_text(conll_prompts()[:10], generation_kwargs)

# annotate text with reference model
annotated_texts = generator.annotate_text(generated_texts)

# save data in correct format
generator.save_data(annotated_texts)

# split dataset
split_data(config.path_data)
