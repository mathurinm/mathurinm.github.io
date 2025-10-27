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
- 11/09		Introduction to GenAI, MLE, Bayes + Base Models, Mixtures of Gaussian, EM ([notes by Y. Dziki](/assets/2025_ens_genmodels/scribe_lecture01.pdf))
- 17/09	  Maximum a posteriori vs max likelihood, PCA, PPCA, VAE [notes by M. Ottavy](/assets/2025_ens_genmodels/scribe_lecture02.pdf)
- at home: LAB 1 : simple generative models: PCA, mixture of Gaussian, pretrained models, VAE
- 18/09	[postponed due to ENS being closed] GAN/WGAN
- 24/09		Flow matching [notes by H. Martel](/assets/2025_ens_genmodels/scribe_lecture03.pdf), [blog post on Flow Matching](https://dl.heeere.com/cfm/)
- 25/09		[Lab on Flow Matching](/assets/2025_ens_genmodels/lab_flow_matching.py)
- 01/10		GANS ([material 1](https://gauthiergidel.github.io/ift_6756_gt_ml/slides/Lecture7.pdf), [material 2](https://gauthiergidel.github.io/ift_6756_gt_ml/slides/Lecture9.pdf), [material 3](https://gauthiergidel.github.io/ift_6756_gt_ml/slides/Lecture11.pdf))
- 02/10		WGAN + [Lab WGAN](/assets/2025_ens_genmodels/lab_wgan.py)
- 08/10		Diffusion 1/2
- 09/10		Introduction to sequence modelling: tokenizers, bigram models, autoregressive models, Transformers
- 15/10		Diffusion 2/2 (links with flow matching, conditional generation)
- 16/10		Metrics (FID/rec/recall/density), conditional generation, classifier/classifier-free guidance, OT, Reflow
- 17/10		**Project progress evaluation**
- 22/10		[Lab transformers](/assets/2025_ens_genmodels/lab_transformers.py)
- 23/10		No class (work on Lab)
- 05/11   Discrete models
- 06/11   Discrete models
- 12/11   Project defense

### Material
- Work in progress: [site for the class](https://generativemodels.github.io/)

### Validation
- weekly homeworks + quizzes + 3 Labs in python
- paper presentation and extension of a selected research article and the associated code applied on real data.


### Prerequisite
- probabilities (densities, change of variable formula)
- linear algebra (PSD matrices, eigenvalue decomposition, spectral theorem)
- calculus (gradient, Hessian, Jacobian, chain rule, ordinary differential equations)
