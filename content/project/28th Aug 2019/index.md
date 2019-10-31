---
title: Softmax
summary: ""
tags:
- Deep Learning
date: "2019-08-28T00:00:00Z"

# Optional external URL for project (replaces project detail page).
external_link: ""

image:
  caption: Softmax
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
Could not sleep because of a jetlag. So I started playing with compiling DL model. In binary classification I use softmax function for probability distribution. However, softmax is not zero centered! The neuron always fires. I am still continuing with Keras and plying with different parameters in fine-tuning my medel. I.e. for optimizer I use "adam" and loss function compute through "class entropy". I am actually thinking I should start using PyTorch, instead NumPy, for running computations on GPU and PySpark (as once Dwight Gunning from FINRA has recommended me while having a meeting).e even hundereds or thousand nodes in a layer (Tensorflow can take care of that).

