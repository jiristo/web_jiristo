---
title: Shared Layers (Keras)
summary: ""
tags:
- Deep Learning
- Inspiration
date: "2019-09-01T00:00:00Z"

# Optional external URL for project (replaces project detail page).
external_link: ""

image:
  caption: "Source: https://mmasucka.com"
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
Learned about **shared layers** allowed by Keras API. They allow me to define an operation and then apply the exact same operation, with the exact same weights on different inputs. Can by used for time-series as well.
I understand, defining multiple input layers for multiple entities (e.g. customer IDs) **allows me to specify** how the data from each entity will be used differently later in this model. Now, I see an applied case in UFC. In data with UFC result for every match and fighter, I can teach the model to learn strength of every single fighter. Such that if any pair of fighters plays each other, I could predict the score, even if those two fighters have never fought before.

