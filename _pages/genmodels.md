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

### Tentative program
<!-- - Intro, analysis reminder, supervised vs generative modeling perspective-->

- 10/09 Class cancelled
- 11/09		Introduction to GenAI, MLE, Bayes + Base Models, Mixtures of Gaussian (EM if time)
- 17/09	Maximum a posteriori vs max likelihood, PCA, PPCA, VAE
- at home: LAB 1 : simple generative models: PCA, mixture of Gaussian, pretrained models, VAE
- 18/09	[cancelled due to ENS being closed] GAN/WGAN
- 24/09		Flow matching
- 25/09		[Lab on Flow Matching](/assets/2025_ens_genmodels/flow_matching_lab.py)
- 01/10		TBD
- 02/10		TBD
- 08/10		Diffusion + link with flow matching
- 09/10		**Project progress evaluation**
- 15/10		Conditional Generative Models
- 16/10		Intro to sequence modeling, tokenizer, base model, Autoregressive models bigrams
- 22/10		Attention, transformers
- 23/10		Evaluation metrics (OT + FID)
- 12/11  Project defense

### Validation
- weekly homeworks + quizzes + 3 Labs in python
- paper presentation and extension of a selected research article and the associated code applied on real data.


### Prerequisite
- probabilities (densities, change of variable formula)
- linear algebra (PSD matrices, eigenvalue decomposition, spectral theorem)
- calculus (gradient, Hessian, Jacobian, chain rule, ordinary differential equations)
