---
title: STOKE playground demo
emoji: 🐢
colorFrom: gray
colorTo: red
sdk: streamlit
sdk_version: 1.31.1
app_file: stoke/playground/app.py
pinned: false
---

# STOKE: A Toolkit for Streaming Token Classification

[Huggingface Space](https://huggingface.co/spaces/nicpopovic/stoke)

[Related publication](https://arxiv.org/abs/2403.11747)

*Note: This code is still being cleaned up currently.*

## Quick start
You can use pip to install the required dependency (including the transformers fork)
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run stoke/playground/app.py
```

This will launch the playground, shown below:

![](stoke/docs/images/playground.png)

## Get custom transformers fork
```
git clone -b STOKE https://github.com/nicpopovic/transformers.git
```