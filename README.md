# STOKE: A Toolkit for Streaming Token Classification

[Huggingface Space](https://huggingface.co/spaces/nicpopovic/stoke)

[Related publication](https://arxiv.org/abs/2403.11747)

## Installation
You can use pip to install the required dependency (including the transformers fork)
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Training Procedure
Example of a data generation and training procedure
```
# first, create an annotated dataset using an auxiliary model
python generate_instruct.py --language_model meta-llama/Llama-3.2-3B-Instruct --cuda

# train probes for layers 20-27, while pointing to the the dataset created above
python example_train.py --path 'data/meta-llama/Llama-3.2-3B-Instruct/STOKE_500_wikiqa' --layers 20 21 22 23 24 25 26 27 --batch_size 4 --cuda
```

## Chat Demo
In order to launch the chat demo (shown below):
```
export HF_TOKEN="your token here..."
python chat.py
```
![](stoke/docs/images/chat_demo.gif)


## Playground
In order to launch the playground (shown below):
```
streamlit run stoke/src/playground/app.py
```

![](stoke/docs/images/playground.png)

## Transformers fork

In order to easily use the streaming classifiers, this repo makes use of a [custom fork of transformers.](https://github.com/nicpopovic/transformers/tree/4.45-STOKE)