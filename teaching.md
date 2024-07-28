---
layout: page
title: Teaching
permalink: /teaching/
---

### Computation Optimal Transport for Machine and Deep Learning

Class details are [here](/otml).


### Optimization for large scale Machine Learning, M2 ENS 2022-2023 & 2023-2024

The goal of the class is to cover theoretical aspects and practical Python implementations of popular optimization algorithms in machine learning, with a focus on modern topics: huge scale models, automatic differentiation, deep learning, implicit bias, etc.

[**Notes for the class are here**]({{ site.baseurl }}{% link /assets/2022_ens/class.pdf  %}){:target="_blank"}{:rel="noopener noreferrer"}.

**Schedule**: From November 21st onwards: Tuesday 08 h 00, Wednesday 13 h 30 (room B1) ****except Wednesday 6th which is moved to Friday 8th****.

**Validation**: some theoretical homeworks, and paper presentation at the end of the class.

**Syllabus**:
- basics of convex analysis: convex sets, convex functions, strong convexity, smoothness, subdifferential, Fenchel-Legendre transform, infimal convolution, Moreau envelope
- gradient descent and subgradient descent, fixed point iterations, proximal point method (Lab 1)
- acceleration of first order methods: Nesterov and momentum
- algorithms for Deep Learning: stochastic gradient descent, ADAM, Adagrad (Lab 2)
- automatic differentiation
- second order algorithms: Newton and quasi-Newton methods
- duality in ML
- implicit regularization, Bregman geometry, mirror descent
- recent results in non convex optimization
- online learning
- other algorithms: Frank-Wolfe, primal-dual algorithms, variational inclusions, extragradient.

Lab 1 on logistic regression is [here](/assets/2022_ens/Lab_logistic_regression.ipynb)

Lab 3 on Deep Learning is [here](/assets/2022_ens/Lab_3_DL_empty.ipynb)

**Resources**:
- _Introductory lectures on convex optimization: a basic course_, Y. Nesterov, 2004. A reference book in optimization, updated in 2018: _Lectures on Convex Optimization_.
- _First order methods in optimization_, A. Beck, 2019.
- _Convex optimization: algorithms and complexity_, S. Bubeck, 2015. A short monograph (100 pages) covering basic topics.


### Classes taught

Summer schools:
- OLISSIPO Winter school: dimensionality reduction with Titouan Vayer (02/2023)
- Convex optimization @Computation and Modelling summer school, WUST 2022 ([intro slides](/assets/2022_wust/slides_intro.pdf) and [exercises](/assets/2022_wust/exos.pdf))

Since my arrival at ENS de Lyon (Nov. 2021):
- 36 h on large scale optimization for machine and deep learning (2022-2024), M2 level.
- 12 h on optimization and approximation (2023-2024), M1 level.

Since 2019, I teach the Python for datascience class (42 h per year) in the X/HEC "Datascience for business" Master, using live coding  inspired by the Software Carpentry workshops. I designed the course from scratch, collaborating  with Joan Massich in 2019, Quentin Bertrand in 2020, Hicham Janati in 2021, Sylvain Combettes in 2022 and Badr Moufad in 2023.

Since 2020 I teach and handle practical sessions and data camps in Ecole Polytechnique's [https://portail.polytechnique.edu/datascience/en/programs/data-science-starter-program-dssp Executive education].
Topics involved dimension reduction, clustering, scaling computations, visualization and datacamp. I designed 2 full python labs with Erwan Le Pennec on these topics.

From 2017 to 2019, as a grad student, my main teaching activity was the Optimization for datascience class of the [Datascience Master](https://www.universite-paris-saclay.fr/formation/master/mathematiques-et-applications/m2-data-sciences), totalling 2*40 h including 4 h as lecturer.
Amongst others, this involved refactoring of the practical sessions, tutoring of students during office hours, and partaking in the design of the final exam.
