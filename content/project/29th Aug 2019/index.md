---
title: Dense Layer (Keras)
summary: ""
tags:
- Deep Learning
date: "2019-08-29T00:00:00Z"

# Optional external URL for project (replaces project detail page).
external_link: ""

image:
  caption: Dense Layer
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
Continuing with DL. Both theory and practice. When it comes to DL network: dense layers learn a **weight matrix**, where the **first dimension of the matrix** is the **dimension of the input data**, and the second dimension is the dimension of the output data.
Defining `tensor` output in one line
```
from keras.layers import Input, Dense
input_tensor = Input(shape=(1,))
output_tensor = Dense(1)(input_tensor)
```
When it comes to loss function, **mean absolute error** is a good general function for a *keras* model. It's less sensitive function to outliers. however, one can use `mse` which would be equivalent to OLS. In DL, $y^{\hat} = mx + b$ where $m$ is the weight of the dense layer and $b$ is the bias of the dense layer. Mean squared error is a common loss function and will optimize for predicting the mean, as is done in OLS. Mean absolute error optimizes for the median and is used in quantile regression.

