---
layout: page
title: Computational Optimal Transport for Machine and Deep Learning
permalink: /otml/
---


In the last decade, optimal transport has rapidly emerged as a versatile tool to compare distributions and clouds of points. As such, it has found numerous successful applications in Statistics, Signal Processing, Machine Learning and Deep Learning.
This class introduces the theoretical and numerical bases of optimal transport, and reviews its latest developments.

## Teachers
Mathurin Massias, Titouan Vayer, Quentin Bertrand.

## Material
- [slides for 1st lecture (20/11/2014)](/assets/2024_ens_ot/course_intro.pdf )
- [course notes on the Simplex algorithm](/assets/2024_ens_ot/simplex.pdf)
- [course notes on Hungarian algorithm](/assets/2024_ens_ot/hungarian.pdf)
- [course notes on entropic optimal transport](/assets/2024_ens_ot/entropic.pdf)
- [slides for domain adaptation lecture (11/12/2014)](/assets/2024_ens_ot/course_da.pdf )
- [slides for Gromov-Wasserstain lecture (19/12/2014)](/assets/2024_ens_ot/course_gromov.pdf)

Homeworks:
- [1st homework](/assets/2024_ens_ot/homework1.pdf) (due 27/11/2024)
- [2nd homework (long one)](/assets/2024_ens_ot/homework2_auction.pdf) (due 20/12/2024)
- [3rd homework](/assets/2024_ens_ot/homework3_bregman.pdf) (due 04/12/2024)
- [lab 1](/assets/2024_ens_ot/lab1.pdf) (done on 05/12/2024)
- [lab 2](/assets/2024_ens_ot/lab2.pdf) (done on 12/12/2024)

## Syllabus
- Discrete optimal transport: optimal assignment, Kantorovich problem, simplex/max flow algorithm
- Duality of OT: C-transforms, dual ascent
- Entropic regularization of OT: Sinkhorn-Knopp algorithm, various interpretations, practical implementations, Sinkhorn divergence
- Applications to domain adaptation, introduction to other regularizations (Laplacian), unbalanced optimal transport and its connexions with sparse optimization
- Applications to color transfer
- Comparing distributions from different spaces: Gromov-Wasserstein and its extensions: Franck Wolfe and entropic regularization, and Bregman algorithms for the Gromov and fused-Gromov-Wasserstein problems
- Application of Gromov-Wasserstein to graphs
- Towards deep learning: how to differentiate through the Wasserstein distance? Introduction to automatic differentation, forward, and backward differentiation of Sinkhorn, implicit diferentiation and Danskin theorem
- Optimal transport for generative modelling: introduction to generative models, Wasserstein generative adversarial networks (WGANs), Wasserstein flows, Schrödinger bridge, optimal transport conditional flow matching, and evaluation of generative models

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
- 50%: Homeworks (small weekly exercises + 3 labs)
- 50%: Paper presentation and extension of a selected research article and the associated code applied on real data.
- possibility to get one bonus point by scribing a lecture (taking notes and typing them in Latex). The associated .tex must be sent by the next Monday to make it available to other students. Rules: only 1 scribe per lecture, max 2 bonus points per person for the whole class.

## Additional ressources
- Computational optimal transport: With applications to data science, G. Peyré and M. Cuturi (2019)
- Tutorial at Neurips 2017: https://media.nips.cc/Conferences/NIPS2017/Eventmedia/nips-2017-marco-cuturi-tutorial.pdf
- Course notes in French: https://perso.math.u-pem.fr/samson.paul-marie/pdf/coursM2transport.pdf
- [Mathurin Massias M2 optimization notes]({{ site.baseurl }}{% link /assets/2022_ens/class.pdf  %}){:target="_blank"}{:rel="noopener noreferrer"}  (especially sections on Convexity, Duality, Gradient descent, Mirror descent)

## Prerequisite
- Differential calculus: gradient, Hessian
- Notions of convexity
- Linear algebra: eigenvalue decomposition, singular value decomposition
