---
title: MLP, AE, RNN
summary: ""
tags:
- Deep Learning
- Recommender System
date: "2019-09-13T00:00:00Z"

# Optional external URL for project (replaces project detail page).
external_link: ""

image:
  caption: "Source: https://commons.wikimedia.org/wiki/File:Recurrent_neural_network_unfold.svg"
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
Returned to Toronto from Europe on Tuesday. I have been recently studying amazing review  of recent efforts on deep learning based recommender systems: **Deep Learning based Recommender System: A Surve and New Perspectives** published by Shuai Zhang et al. So why DL based recsys? Firstly, DL models are end-to-end differentiable, secondly they provide suitable inductive biases. Therefore, DL models can exploit inherent structures if there are any.
I particularly like MLP for its simplicity and capability to learn hierarchical features.
AE are oslo used in DL based recsys as unsupervised model reducing dimensionality in a similar manner as PCA. AE can be used both for for item-based and user-based models.

AE is also powerful in learning feature representation, so it is possible to learn feature representations for user/item content features.
Another interesting type of AE is Collaborative Deep Learning (CDL). It is a hierarchical model Bayesian model with stacked denoising AE (SDEAE). CDL has two components: i.) perception component (deep neural net.: probabilistic interpretation of of ordinal SDAE ) and ii.) task-specific component (PMF).
Collaborative Deep Learning is a pairwise framework for top-n recommendations.  Research shows that it can even outperform CDL when it comes to ranking predictions. It outputs a confidence value $C^{-1}_{uij}$ of how much a user *u* preferes item *i* over *j*. 
 
 RNN also get my attention becase I see its usage in time-series because of variants such as LSTM. More interestingly, DRL is exciting because of a possibility to make recommendations based on on trial-and-error paradigm. 
For sequential modeling,  RNN (interval memory)  and CNN (filters sliding along with time, i.e. kernel). Sequential signals are important for mining temporal dynamics of user behaviour and time evolution. 

When it comes to activation functions, **ReLu should be one to go for!** Unlike in **sigmoid** where vanishing gradient may occur and its output is not zero centered, ReLu is always positive with the slope of 1. If the neurons die, one may use leaky ReLu which output may be even negative (but close to 0!). **Tanh** gives a zero centered output but again, vanishing gradient may arise as well. 
