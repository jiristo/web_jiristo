---
title: Activation Function ReLu
summary: ""
tags:
- Deep Learning
date: "2019-08-27T00:00:00Z"

# Optional external URL for project (replaces project detail page).
external_link: ""

image:
  caption: ReLu
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
Activation function: Inputs (variables or outputs of neurons) get weights. If the **dot product** of these weights is higher than certain value, the activation function simulata if the neuros fire or not. Activation function is close to a sigmoid function (e.g. square root function), so the output is between 0 and 1. **Activation function captures non-linearities**. Classification: Dog (0.71) or cat(0.29), i.e. 71% confidence the object is a dog. So why to use DL instead of linear regression? DL models allow interactions (no assumption of no multicollinearity). Cool thing about NNs is that I don't need to specify interactions. They are captured by the model itself. ReLu is actually very simple:

```
def relu(input):
    '''Defines ReLu activation function: Input node dot product'''
    # Calculate the value for the output of the relu function: output
    output = max(0, input)

    # Return the value just calculated
    return(output)
```
e.g.
```
print(relu(5))
#5
print(relu(-10))
#0
```

Deep networks internally build representations of patterns in the data. Partially replace the need for feature engineering. Subsequent layers build increasingly sophisticated
representations of raw data. Gradient descent is loss vs weight (I want to minimize the loss). Learned about back propagation. I need to recap it once (or probably several times) more. At least, I understand the analogy between a neural network  and brain neuron. It fires back and forward! DC has a great tutorial by Dan Becker (Data Scientist at at Google).

Keras: Dense layer means all the nodes in the current layer connect to all the nodes in the previous layer. There may be even hundereds or thousand nodes in a layer (Tensorflow can take care of that).

