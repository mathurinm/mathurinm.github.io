---
layout: page
title: Computational Optimal Transport for Machine and Deep Learning
permalink: /otml/
---


In the last decade, optimal transport has rapidly emerged as a versatile tool to compare distributions and clouds of points. As such, it has found numerous successful applications in Statistics, Signal Processing, Machine Learning and Deep Learning.
This class introduces the theoretical and numerical bases of optimal transport, and reviews its latest developments.

## Teachers
Mathurin Massias, Titouan Vayer, Quentin Bertrand.

## Syllabus
- Discrete optimal transport: optimal assignment, Kantorovich problem, simplex/max flow algorithm
- Duality of OT: C-transforms, dual ascent,
- Entropic regularization of OT: Sinkhorn algorithm, interpretation as mirror descent, practical implementations
- Applications to domain adaptation, introduction to other regularizations (Laplacian), unbalanced optimal transport and its connexions with sparse optimization
- Applications to color transfer
- Comparing distributions from different spaces: Gromov-Wasserstein and its extensions: Franck Wolfe and entropic regularization, and Bregman algorithms for the Gromov and fused-Gromov-Wasserstein problems
- Application of Gromov-Wasserstein to graphs
- Towards deep learning: how to differentiate through the Wasserstein distance? Introduction to automatic differentation, forward, and backward differentiation of Sinkhorn, implicit diferentiation and Danskin theorem
- Optimal transport for generative modelling: introduction to generative models, Wasserstein generative adversarial networks (WGANs), Wasserstein flows, Schödinger bridge, optimal transport conditional flow matching, and evaluation of generative models

- Potential extensions: learning Monge maps, Wasserstein spaces, sliced Wasserstein, etc.
<!-- #- Brenier?
#!-- - Wasserstein spaces, Wasserstein barycenters
#- sliced Wasserstein
#- Statistical view of OT
#- Gromov, fused? -->

## Schedule
15 x 2 h of class/labs, oral presentation

Classes on Wednesday 15 h 45 (room 029) and Thursday 15 h 45 (room B1), the following weeks:
Nov 18 - 25, Dec 2 - 9 - 16, Jan 6 - 13 - 20 -27

## Validation
Paper presentation and extension of a selected research article and the associated code applied on real data.

## Ressources
Computational optimal transport: With applications to data science, G. Peyré and M. Cuturi (2019)


## Prerequisite
- Differential calculus: gradient, Hessian
- Notions of convexity
- Linear algebra: eigenvalue decomposition, singular value decomposition
