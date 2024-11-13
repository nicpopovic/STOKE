import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, STOKEStreamer
from threading import Thread
import json
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from bs4 import BeautifulSoup


def clean_html(html_content):
    # Parse the HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove all elements with class 'small-text'
    for element in soup.find_all(class_='small-text'):
        element.decompose()  # Removes the element from the tree

    # Get the plain text, stripping any remaining HTML tags
    cleaned_text = soup.get_text()

    return cleaned_text.strip().replace("  ", " ").replace("( ", "(").replace(" )", ")")

# Reusing the original MLP class and other functions (unchanged) except those specific to Streamlit
class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024, layer_id=0, cuda=False):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)
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
    value = np.clip(value, 0.0, 1.0)
    colormap = plt.get_cmap(colormap_name)
    rgba_color = colormap(value)
    css_color = to_hex(rgba_color)
    return css_color + "88"

# Caching functions for model and classifier
model_cache = {}

def get_model_and_tokenizer(name):
    if name not in model_cache:
        tok = AutoTokenizer.from_pretrained(name, token=os.getenv("HF_TOKEN"))
        model = AutoModelForCausalLM.from_pretrained(name, token=os.getenv("HF_TOKEN"), torch_dtype="bfloat16")
        model_cache[name] = (model, tok)
    return model_cache[name]

def get_classifiers_for_model(att_size, emb_size, device, config_paths):
    config = {
        "classifier_token": json.load(open(os.path.join(config_paths["classifier_token"], "config.json"), "r")),
        "classifier_span": json.load(open(os.path.join(config_paths["classifier_span"], "config.json"), "r"))
    }
    layer_id = config["classifier_token"]["layer"]
    
    classifier_span = MLP(att_size, 2, hidden_dim=config["classifier_span"]["classifier_dim"]).to(device)
    classifier_span.load_state_dict(torch.load(os.path.join(config_paths["classifier_span"], "checkpoint.pt"), map_location=device))

    classifier_token = MLP(emb_size, len(config["classifier_token"]["label_map"]), layer_id=layer_id, hidden_dim=config["classifier_token"]["classifier_dim"]).to(device)
    classifier_token.load_state_dict(torch.load(os.path.join(config_paths["classifier_token"], "checkpoint.pt"), map_location=device))

    return classifier_span, classifier_token, config["classifier_token"]["label_map"]

def find_datasets_and_model_ids(root_dir):
    datasets = {}
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


def generate_html_spanwise(token_strings, tokenwise_preds, spans, tokenizer, new_tags):

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
                annotated.append(" ")

            span_opener = f" <span class='{predicted_class}' data-tooltip-text='{new_tags[tokenwise_preds[i]]}' style='{style}'>".replace(" ", " ")
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

# Creating the Gradio Interface
def generate_text(input_text, messages=None):
    if input_text == "":
        yield "Please enter some text first."
        return
    
    token_limit=350
    #print([clean_html(x["content"]) for x in messages])
    
    streamer = STOKEStreamer(tok, classifier_token, classifier_span)

    new_tags = label_map
    
    if messages is None:
        messages = []
    else:
        messages = []
    system="""You are a knowledge assistant. Keep your responses very short."""
    messages = [{"role": "system", "content": system}]+ [{"role": x["role"], "content": clean_html(x["content"])} for x in messages] +[{"role": "user", "content": input_text}]
    input_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tok([input_text], return_tensors="pt").to(model.device)

    if len(inputs.input_ids[0]) > 80:
        yield [{"role": "assistant", "content": "Your message is too long for this demo, sorry :("}]
        return

    #inputs = tok([f"  {input_text[:200]}"], return_tensors="pt").to(model.device)
    #inputs = tok([input_text[:200]], return_tensors="pt").to(model.device)
    generation_kwargs = dict(
        inputs, streamer=streamer, max_new_tokens=token_limit-len(inputs.input_ids[0]), 
        repetition_penalty=1.2, do_sample=False
    )

    def generate_async():
        model.generate(**generation_kwargs)

    thread = Thread(target=generate_async)
    thread.start()

    # Display generated text as it becomes available
    output_text = ""
    text_tokenwise = ""
    text_spans = ""
    removed_spans = ""
    tags = []
    spans = []
    for new_text in streamer:
        if new_text[1] is not None and new_text[2] != ['']:
            text_tokenwise = ""
            output_text = ""
            tags.extend(new_text[1])
            spans.extend(new_text[-1])

            # Tokenwise Classification
            for tk, pred in zip(new_text[2],tags):
                if pred != 0:
                    style = f"background-color: {map_value_to_color((pred-1)/(len(new_tags)-1))}"
                    if tk.startswith(" "):
                        text_tokenwise += " "
                    text_tokenwise += f"<span class='tooltip highlight' data-tooltip-text='{new_tags[pred]}' style='{style}'>{tk}</span>"
                    output_text += tk
                else:
                    text_tokenwise += tk
                    output_text += tk

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
            annotated_tokens = generate_html_spanwise(new_text[2], tags, [x for x in filter_spans(spans)[0]], tok, new_tags)
            #generated_text_spanwise = tok.convert_tokens_to_string(annotated_tokens).replace("<|endoftext|>", "").replace("<|begin_of_text|>", "")
            generated_text_spanwise = "".join(annotated_tokens).replace("<|endoftext|>", "").replace("<|begin_of_text|>", "")

            output = generated_text_spanwise
            #output += "<h5>Show tokenwise classification</h5>\n" + text_tokenwise.replace("\n", " ").replace("$", "\\$").replace("<|endoftext|>", "").replace("<|begin_of_text|>", "")
            #output += "</details><details><summary>Show spans</summary>\n" + text_spans.replace("\n", " ").replace("$", "\\$")
            #if removed_spans != "":
            #    output += f"<br><br><i>({removed_spans})</i>"
            list_of_spans = [{"name": tok.convert_tokens_to_string(new_text[2][x[0]:x[1]]).strip(), "type": new_tags[tags[x[1]-1]]} for x in filter_spans(spans)[0] if new_tags[tags[x[1]-1]] != "O"]

            out_dict = {"text": output_text.replace("<|endoftext|>", "").replace("<|begin_of_text|>", "").strip(), "entites": list_of_spans}
            
            if output.endswith("<|end_header_id|>\n\n"):
                continue
            html_out = output.replace("<|endoftext|>", "").replace("<|begin_of_text|>", "").strip().split("<|end_header_id|>")[-1].replace("**", "")

            yield [messages[-1]] + [{"role": "assistant", "content": html_out}]

    return

# Load datasets and models for the Gradio app
datasets = find_datasets_and_model_ids("data/")
available_models = list(datasets.keys())
available_datasets = {model: list(datasets[model].keys()) for model in available_models}
available_configs = {model: {dataset: list(datasets[model][dataset].keys()) for dataset in available_datasets[model]} for model in available_models}

def update_datasets(model_name):
    return available_datasets[model_name]

def update_configs(model_name, dataset_name):
    return available_configs[model_name][dataset_name]

model_id = "meta-llama/Llama-3.2-1B-Instruct"
data_id = "STOKE_500_wikiqa"
config_id = "default"

#model_id = "gpt2"
#data_id = "1_NER"
#config_id = "default"

model, tok = get_model_and_tokenizer(model_id)
if torch.cuda.is_available():
    model.cuda()

# Load model classifiers
try:
    classifier_span, classifier_token, label_map = get_classifiers_for_model(
        model.config.n_head * model.config.n_layer, model.config.n_embd, model.device,
        datasets[model_id][data_id][config_id]
    )
except:
    classifier_span, classifier_token, label_map = get_classifiers_for_model(
        model.config.num_attention_heads * model.config.num_hidden_layers, model.config.hidden_size, model.device,
        datasets[model_id][data_id][config_id]
    )


css = """
    <style>
    .prose {
        line-height: 200%;
    }
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
    color: grey!important;
}
footer {
    display:none !important
    } 
    .gradio-container {
        padding: 0!important;
        height:400px;
        }
    </style>"""
"""
with gr.Blocks(css=css, elem_id="chatbox") as demo:
    gr.ChatInterface(generate_text, examples=["Who where the Beatles?", "Whats the GDP of Norway?", "List some fun things to do in Miami", "What do you know about the KIT in Karlsruhe?", "Give me a list of the most iconic 90s songs", "Whats the typical cost of a pizza in New York City?", "Got any suggestions for a day trip from Miami?", "Tell me about the climate in Europe.", "Where can I go scuba diving?", "give me a list of famous people and their years of birth"], type="messages")
"""

example_messages=[{'role': 'user', 'content': "I'm going to Miami. What should I do there?"}, {'role': 'assistant', 'content': """<span class='highlight spanhighlight' data-tooltip-text='GPE' style='background-color: #e6550d88'>Miami<span class='small-text'>GPE</span></span> has plenty of exciting activities:

* Visit <span class='highlight spanhighlight' data-tooltip-text='LOC' style='background-color: #fdd0a288'> South Beach<span class='small-text'>LOC</span></span> for art deco architecture and vibrant nightlife.
* Explore the  <span class='highlight spanhighlight' data-tooltip-text='FAC' style='background-color: #c6dbef88'> Vizcaya Museum & Gardens<span class='small-text'>FAC</span></span>, an estate with European-inspired gardens.
* Take a stroll through  <span class='highlight spanhighlight' data-tooltip-text='FAC' style='background-color: #c6dbef88'> Little Havana's Calle Ocho<span class='small-text'>FAC</span></span> ( <span class='highlight spanhighlight' data-tooltip-text='FAC' style='background-color: #c6dbef88'>8th Street<span class='small-text'>FAC</span></span>) for  <span class='highlight spanhighlight' data-tooltip-text='NORP' style='background-color: #a1d99b88'> Cuban<span class='small-text'>NORP</span></span> culture and food.
* Relax on  <span class='highlight spanhighlight' data-tooltip-text='LOC' style='background-color: #fdd0a288'> Miami Beach<span class='small-text'>LOC</span></span> or visit  <span class='highlight spanhighlight' data-tooltip-text='FAC' style='background-color: #c6dbef88'> Crandon Park<span class='small-text'>FAC</span></span> for snorkeling and beach activities.

Which one interests you most?"""}]

with gr.Blocks(css=css, fill_width=True) as demo:
    chatbot = gr.Chatbot(type="messages", value=example_messages)
    msg = gr.Textbox(submit_btn=True, max_length=80, placeholder="Type your message here...")
    msg.submit(lambda: None, None, chatbot).then(generate_text, msg, chatbot, queue="queue")
    # Add an examples section for users to pick from predefined messages
    examples = gr.Examples(examples=["What can you tell me about the Beatles?", "Whats the GDP of Norway?", "I'm going to Miami. What should I do there?", "What do you know about the KIT in Karlsruhe?"], inputs=msg, run_on_click=True, fn=generate_text, outputs=chatbot)




demo.launch()

