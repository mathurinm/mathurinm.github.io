---
layout: page
title: Generative models
permalink: /genmodels/
---


# Generative modelling

In the wake of models like Dall-E and ChatGPT, generative models have had a massive impact on text and image applications. The goal of this class is to present their mathematical and algorithmical foundations.


<figure>
  <div class="l-page" style="--ar: calc(218 / 161)">
    <iframe style="aspect-ratio: 1; width: calc(100% / ( 1 + 2 * var(--ar)));" src="{{ 'assets/2025_ens_genmodels/ot-flow-1d.html#loop9' | relative_url }}" frameborder="0" scrolling="no"></iframe>
    <img style="vertical-align: top; width: calc(100% * var(--ar) / ( 1 + 2 * var(--ar)));" src="{{ 'assets/2025_ens_genmodels/pbackground.svg' | relative_url }}" />
    <iframe style="aspect-ratio: var(--ar); margin: 0 -5px; width: calc(100% * var(--ar) / ( 1 + 2 * var(--ar)));" src="{{ 'assets/2025_ens_genmodels/u-anim.html' | relative_url }}" frameborder="0" scrolling="no"></iframe>
  </div>
</figure>


### Teachers:
- Quentin Bertrand (CR Inria, MALICE team)
- Rémi Emonet (IUF, Associate professor, Jean Monnet University, MALICE Team)
- Mathurin Massias (CR Inria, OCKHAM team)

### Syllabus
<!-- - Intro, analysis reminder, supervised vs generative modeling perspective-->

Generative modelling for images:
- Normalizing flows, Continuous normalizing flows
- Flow matching
- Diffusion models (DDPM, DDIM, Stochastic differential equations), latent diffusion
- GANs

Models for text:
- Autoregressive models, text-to-text models
- Transformers (the ChatGPT architecture)
- Mamba and State space models
- Text-to-image, conditional generation

Additional topics:
- Variational inference
- Sampling as optimization/energy-based model
- Evaluation of Generative Models

### Validation
- 50 % weekly homeworks + 3 Labs in python
- 50 % paper presentation and extension of a selected research article and the associated code applied on real data.


### Prerequisite
- probabilities (densities, change of variable formula)
- linear algebra (PSD matrices, eigenvalue decomposition, spectral theorem)
- calculus (gradient, Hessian, Jacobian, chain rule, ordinary differential equations)
