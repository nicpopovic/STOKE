import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, STOKEStreamer
from threading import Thread
import json
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import numpy as np
import os
import urllib.request
import zipfile


class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024, layer_id=0, cuda=False):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)  # Input layer to hidden layer
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)  # Hidden layer to output layer
        self.layer_id = layer_id
        if cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.to(self.device)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc3(x)
        
        return torch.argmax(x, dim=-1).cpu().detach(), torch.softmax(x, dim=-1).cpu().detach()

def map_value_to_color(value, colormap_name='tab20c'):
    """
    Map a value between 0 and 1 to a CSS color using a Python colormap.

    Args:
        value (float): A value between 0 and 1.
        colormap_name (str): The name of the colormap to use (e.g., 'viridis').

    Returns:
        str: A CSS color string in the form 'rgb(r, g, b)'.
    """
    # Ensure the value is within the range [0, 1]
    value = np.clip(value, 0.0, 1.0)

    # Get the colormap
    colormap = plt.get_cmap(colormap_name)

    # Map the value to a color
    rgba_color = colormap(value)

    # Convert the RGBA color to CSS format
    css_color = to_hex(rgba_color)

    return css_color + "88"

@st.cache_resource
def get_model_and_tokenizer(name):
    # Load pre-trained model and tokenizer
    tok = AutoTokenizer.from_pretrained(name, token=os.getenv("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(name, token=os.getenv("HF_TOKEN"))
    return model, tok

@st.cache_resource
def get_classifiers_for_model(att_size, emb_size, device, config_paths):
    classifier_token = None
    #print(config)
    config = {
        "classifier_token": json.load(open(os.path.join(config_paths["classifier_token"], "config.json"), "r")),
        "classifier_span": json.load(open(os.path.join(config_paths["classifier_span"], "config.json"), "r"))
    }

    layer_id = config["classifier_token"]["layer"]
    
    classifier_span = MLP(att_size, 2, hidden_dim=config["classifier_span"]["classifier_dim"]).to(device)
    classifier_span.load_state_dict(torch.load(os.path.join(config_paths["classifier_span"], "checkpoint.pt"), map_location=device))

    classifier_token = MLP(emb_size, len(config["classifier_token"]["label_map"]), layer_id=layer_id, hidden_dim=config["classifier_token"]["classifier_dim"]).to(device)
    classifier_token.load_state_dict(torch.load(os.path.join(config_paths["classifier_token"], "checkpoint.pt"), map_location=device))

    print(sum(p.numel() for p in classifier_span.parameters()), sum(p.numel() for p in classifier_token.parameters()))

    return classifier_span, classifier_token, config["classifier_token"]["label_map"]

def get_available_models():
    available_models = []
    for model_name in ["gpt2", "gpt2-xl"]:
        if os.path.isfile(f"checkpoints/{model_name}/config.json"):
            available_models.append(model_name)
    return available_models

def get_available_datasets(model_name):
    available_datasets = []
    config_path = f"checkpoints/{model_name}/config.json"
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            # Assuming datasets are keys in config.json
            available_datasets = list(config.keys())
    return available_datasets

def download_and_extract_zip(url, extract_dir):
    # Determine the parent directory
    parent_dir = os.path.split(os.path.dirname(extract_dir))[-2]
    print(parent_dir)
    
    # Download the zip file to the parent directory
    zip_file_path = os.path.join(parent_dir, "data.zip")
    urllib.request.urlretrieve(url, zip_file_path)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(parent_dir)
    
    # Remove the zip file
    os.remove(zip_file_path)

def find_datasets_and_model_ids(root_dir):
    datasets = {}
    
    # Check if the root directory exists
    if not os.path.exists(root_dir):
        # If root directory doesn't exist, download a zip file and unpack it
        print("Root directory doesn't exist. Downloading zip file...")
        url = "https://drive.usercontent.google.com/download?id=1i5UkWikRZGhsbv21ZZSjEZl6-VwNC0lp&export=download&authuser=0&confirm=t&uuid=c33ef625-9ec8-4dbf-bdb0-ad6cabc70a33&at=APZUnTWWJSzU9pV2XV-sMPtbgdgj%3A1711096726305"  # Replace with your actual download URL
        download_and_extract_zip(url, root_dir)
        print("Zip file downloaded and unpacked successfully.")

    
    for root, dirs, files in os.walk(root_dir):
        if 'config.json' in files and 'stoke_config.json' in files:
            config_path = os.path.join(root, 'config.json')
            stoke_config_path = os.path.join(root, 'stoke_config.json')

            with open(config_path, 'r') as f:
                config_data = json.load(f)
                model_id = config_data.get('model_id')
                if model_id:
                    dataset_name = os.path.basename(os.path.dirname(config_path))

            with open(stoke_config_path, 'r') as f:
                stoke_config_data = json.load(f)
                if model_id:
                    dataset_name = os.path.basename(os.path.dirname(stoke_config_path))
                    datasets.setdefault(model_id, {})[dataset_name] = stoke_config_data
    
    return datasets


# Main content
st.title("Playground")

# Sidebar for model and dataset selection
with st.sidebar:
    st.subheader("Model and Dataset Selection")
    datasets = find_datasets_and_model_ids("data/")
    available_models = datasets.keys()
    print(datasets)
    if available_models:
        model_selection = st.selectbox("Select Model", available_models)
    else:
        st.error("No models available. Please check the file paths.")

    # Select dataset based on selected model
    available_datasets = datasets[model_selection]
    if available_datasets:
        dataset_selection = st.selectbox("Select Dataset", available_datasets)
    else:
        st.error("No datasets available for the selected model.")

    # Select dataset based on selected model
    available_configs = datasets[model_selection][dataset_selection]
    if available_configs:
        config_selection = st.selectbox("Select Config", available_configs.keys())
    else:
        st.error("No configs available for the selected dataset.")

# Load model and streamer based on selections
model, tok = get_model_and_tokenizer(model_selection)
if torch.cuda.is_available():
    model.cuda()
classifier_span, classifier_token, label_map = get_classifiers_for_model(model.config.num_attention_heads*model.config.num_hidden_layers, model.config.hidden_size, model.device, datasets[model_selection][dataset_selection][config_selection])
streamer = STOKEStreamer(tok, classifier_token, classifier_span)

new_tags = label_map


def filter_spans(spans_and_values):
    if spans_and_values == []:
        return [], []
    # Create a dictionary to store spans based on their second index values
    span_dict = {}

    spans, values = [x[0] for x in spans_and_values], [x[1] for x in spans_and_values]

    # Iterate through the spans and update the dictionary with the highest value
    for span, value in zip(spans, values):
        start, end = span
        if start > end or end - start > 15 or start == 0:
            continue
        current_value = span_dict.get(end, None)

        if current_value is None or current_value[1] < value:
            span_dict[end] = (span, value)

    if span_dict == {}:
        return [], []
    # Extract the filtered spans and values
    filtered_spans, filtered_values = zip(*span_dict.values())

    return list(filtered_spans), list(filtered_values)

def remove_overlapping_spans(spans):
    # Sort the spans based on their end points
    sorted_spans = sorted(spans, key=lambda x: x[0][1])
    
    non_overlapping_spans = []
    last_end = float('-inf')
    
    # Iterate through the sorted spans
    for span in sorted_spans:
        start, end = span[0]
        value = span[1]
        
        # If the current span does not overlap with the previous one
        if start >= last_end:
            non_overlapping_spans.append(span)
            last_end = end
        else:
            # If it overlaps, choose the one with the highest value
            existing_span_index = -1
            for i, existing_span in enumerate(non_overlapping_spans):
                if existing_span[0][1] <= start:
                    existing_span_index = i
                    break
            if existing_span_index != -1 and non_overlapping_spans[existing_span_index][1] < value:
                non_overlapping_spans[existing_span_index] = span
    
    return non_overlapping_spans

def generate_html_no_overlap(tokenized_text, spans):
    current_index = 0
    html_content = ""

    for (span_start, span_end), value in spans:
        # Add text before the span
        html_content += "".join(tokenized_text[current_index:span_start])

        # Add the span with underlining
        html_content += "<b><u>"
        html_content += "".join(tokenized_text[span_start:span_end])
        html_content += "</u></b> "

        current_index = span_end

    # Add any remaining text after the last span
    html_content += "".join(tokenized_text[current_index:])

    return html_content


css = """
    <style>
    .highlight {
        display: inline;
    }
    .highlight::after {
        background-color: var(data-color);
    }
    .spanhighlight {
        padding: 2px 5px;
        border-radius: 5px;
    }
    .tooltip {
    position: relative;
    display: inline-block;
}

.tooltip::after {
    content: attr(data-tooltip-text); /* Set content from data-tooltip-text attribute */
    display: none;
    position: absolute;
    background-color: #333;
    color: #fff;
    padding: 5px;
    border-radius: 5px;
    bottom: 100%; /* Position it above the element */
    left: 50%;
    transform: translateX(-50%);
    width: auto;
    min-width: 120px;
    margin: 0 auto;
    text-align: center;
}

.tooltip:hover::after {
    display: block; /* Show the tooltip on hover */
}

.small-text {
    padding: 2px 5px;
    background-color: white;
    border-radius: 5px;
    font-size: xx-small;
    margin-left: 0.5em;
    vertical-align: 0.2em;
    font-weight: bold;
    color: grey;
}
    </style>"""


def generate_html_spanwise(token_strings, tokenwise_preds, spans, tokenizer):

    # spanwise annotated text
    annotated = []
    span_ends = -1
    in_span = False

    out_of_span_tokens = []
    for i in reversed(range(len(tokenwise_preds))):

        if in_span:
            if i >= span_ends:
                continue
            else:
                in_span = False

        predicted_class = ""
        style = ""

        span = None
        for s in spans:
            if s[1] == i+1:
                span = s

        if tokenwise_preds[i] != 0 and span is not None:
            predicted_class = f"highlight spanhighlight"
            style = f"background-color: {map_value_to_color((tokenwise_preds[i]-1)/(len(new_tags)-1))}"
            if tokenizer.convert_tokens_to_string([token_strings[i]]).startswith(" "):
                annotated.append("Ġ")

            span_opener = f"Ġ<span class='{predicted_class}' data-tooltip-text='{new_tags[tokenwise_preds[i]]}' style='{style}'>".replace(" ", "Ġ")
            span_end = f"<span class='small-text'>{new_tags[tokenwise_preds[i]]}</span></span>"
            annotated.extend(out_of_span_tokens)
            out_of_span_tokens = []
            span_ends = span[0]
            in_span = True
            annotated.append(span_end)
            annotated.extend([token_strings[x] for x in reversed(range(span[0], span[1]))])
            annotated.append(span_opener)
        else:
            out_of_span_tokens.append(token_strings[i])

    annotated.extend(out_of_span_tokens)

    return [x for x in reversed(annotated)]

# Define function to generate text based on input
def generate_text(generation_kwargs, output_field):
    
    # Function to generate text in a separate thread
    def generate_async():
        model.generate(**generation_kwargs)
    
    # Start text generation in a separate thread
    thread = Thread(target=generate_async)
    thread.start()

    # Display generated text as it becomes available
    text_tokenwise = ""
    text_spans = ""
    removed_spans = ""
    tags = []
    spans = []
    for new_text in streamer:
        if new_text[1] is not None and new_text[2] != ['']:
            text_tokenwise = ""
            tags.extend(new_text[1])
            spans.extend(new_text[-1])

            # Tokenwise Classification
            for tk, pred in zip(new_text[2],tags):
                if pred != 0:
                    style = f"background-color: {map_value_to_color((pred-1)/(len(new_tags)-1))}"
                    if tk.startswith(" "):
                        text_tokenwise += " "
                    text_tokenwise += f"<span class='tooltip highlight' data-tooltip-text='{new_tags[pred]}' style='{style}'>{tk}</span>"
                else:
                    text_tokenwise += tk

            # Span Classification
            text_spans = ""
            if len(spans) > 0:
                filtered_spans = remove_overlapping_spans(spans)
                text_spans = generate_html_no_overlap(new_text[2], filtered_spans)
                if len(spans) - len(filtered_spans) > 0:
                    removed_spans = f"{len(spans) - len(filtered_spans)} span(s) hidden due to overlap."
            else:
                for tk in new_text[2]:
                    text_spans += f"{tk}"

            # Spanwise Classification
            annotated_tokens = generate_html_spanwise(new_text[2], tags, [x for x in filter_spans(spans)[0]], tok)
            generated_text_spanwise = tok.convert_tokens_to_string(annotated_tokens).replace("<|endoftext|>", "").replace("<|begin_of_text|>", "")

            output_field.empty()
            output = f"{css}"
            output += generated_text_spanwise.replace("\n", " ").replace("$", "$") + "\n<br>"
            output += "<details><summary>Tokenwise classification</summary>\n" + text_tokenwise.replace("\n", " ").replace("$", "\\$").replace("<|begin_of_text|>", "")
            #output += "</details><details><summary>Show spans</summary>\n" + text_spans.replace("\n", " ").replace("$", "\\$")
            #if removed_spans != "":
            #    output += f"<br><br><i>({removed_spans})</i>"
            output += "</details>"
            output_field.write(output, unsafe_allow_html=True)

# Input field
input_text = st.text_area("Enter prompt for completion", "")

# Sidebar for customizing generation parameters
with st.sidebar:
    st.subheader("Generation Parameters")
    max_new_tokens = st.slider("Max New Tokens", min_value=1, max_value=500, value=30)
    repetition_penalty = st.slider("Repetition Penalty", min_value=1.0, max_value=2.0, value=1.2)
    do_sample = st.checkbox("Do Sample", value=True)
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0)
    top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=0.3)
    top_k = st.slider("Top-k", min_value=10, max_value=100, value=50)
    typical_p = st.slider("Typical P", min_value=0.1, max_value=1.0, value=1.0)

# Button to generate text
if st.button("Generate"):
    if input_text:
        output_field = st.empty()
        inputs = tok(["  " + input_text], return_tensors="pt").to(model.device)
        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_new_tokens, 
                                 repetition_penalty=repetition_penalty, temperature=temperature, 
                                 top_p=top_p, top_k=top_k, do_sample=do_sample, typical_p=typical_p)
        generate_text(generation_kwargs, output_field)
    else:
        st.warning("Please enter some text first.")
