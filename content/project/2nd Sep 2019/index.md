---
title: Embedding Layers & Shared Models (Keras)
summary: ""
tags:
- Deep Learning
date: "2019-09-02T00:00:00Z"

# Optional external URL for project (replaces project detail page).
external_link: ""

image:
  caption: "Source: https://www.kdnuggets.com/2017/02/stacking-models-imropved-predictions.html"
  focal_point: Smart

links:
- icon: twitter
  icon_pack: fab
  name: Follow
  url: https://twitter.com/jiristo
url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
slides: example
---
Keras models can have both trainable and fixed parameters. Fewer trainable parameters are less flexibel but at the same time less likely to overfit. Trainable parameters are usually in **dense** layers. Due to an embedding layer in the model (input) and dense layer (with e.g. 4 parameters), the model has much more trainable parameters. **Embedding layers often add a large number of trainable parameters into the model** (be cautious of overfitting). Embedding layer maps integers to floats: each unique value of of the embedding input gets a parameter for its output.
**Shared models** work the same way as shared layers, i.e. I can put together a sequence of layers to define a custom model. Then, it's possible to share the entire model in exactly the same way I would share a layer.
**Model stacking** is using model's prediction as input to another model. It's one of the most sophisticated way of combining models. When stacking it's important to use different datasets for each model.
It's also possible to create single model for classification and regression with Keras. In ordet to do that, I use `activation = "sigmoid"` in output layer. With two outputs model's each output requires its own loss function passed as a list (e.g. different loss functions for a classification and regression models). Optimzer, e.g. `"adam"`, can be passed only once for the both model

