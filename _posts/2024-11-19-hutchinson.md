---
layout: post
title:  "The Hutchinson trick"
date:   2024-11-19 00:00:00 +0200
permalink: /:title/
---

The Hutchinson trick: a cheap way to evaluate the trace of a Jacobian, without computing the Jacobian itself!

<div class="preamble">
$$
\DeclareMathOperator{\DIV}{div}
\DeclareMathOperator{\tr}{tr}
\DeclareMathOperator{\jac}{J}
\newcommand{\Id}{\mathrm{Id}}
\newcommand{\bbR}{\mathbb{R}}
\newcommand{\bbE}{\mathbb{E}}
\DeclareMathOperator{\Var}{Var}
\DeclareMathOperator{\diag}{diag}
\newcommand{\norm}[1]{\Vert #1 \Vert}
$$
</div>


Continuous normalizing flows (more in a blog post to come) heavily rely on the computation of the divergence of a network $$f: \bbR^d \to \bbR^d$$, aka the trace of its Jacobian:

$$
\begin{equation}
\DIV f (x) = \tr \jac_f(x) \triangleq \sum_{i=1}^d \frac{\mathrm{d} f_i}{\mathrm{d} x_i} (x)
\end{equation}
$$

Computing this quantity would require evaluating the full Jacobian, which can be done at the prohibitive cost of $$d$$ backpropagations (more details below).
It turns out, the trace/divergence can be approximated reasonably well **with a single call to backpropagation**.

For that, let's forget about Jacobians for a second and take a generic matrix $$A \in \bbR^{d \times d}$$.
The Hutchinson trick states that for any random variable $$z \in \bbR^d$$ such that $$\bbE[zz^\top] = \Id_d$$,
$$
\begin{equation}\label{eq:hutchinson}
\tr(A) = \bbE_z [z^\top A z]
\end{equation}
$$

It is typically used for $$z$$ having iid entries of mean zero and variance, classically standard Gaussian or Rademacher.
The proof is very simple:

$$
\begin{align*}
\bbE_z [z^\top A z]
&= \bbE_z [\tr(z^\top A z)]  \quad &\text{(a scalar equals its trace)} \\
&= \bbE_z [\tr (A z z^\top)] \quad  &(\tr(MN) = \tr(NM))  \\
&= \tr \left(\bbE_z [A z z^\top]\right)  \quad &\text{(trace and expectation commute)}  \\
&= \tr \left(A \bbE_z [z z^\top] \right)  \quad &\text{(linearity of expectation)}  \\
&= \tr \left(A \Id_d \right)  \quad &\text{(assumption on $z$ )}  \\
&= \tr \left(A  \right)  \quad &
\end{align*}
$$

et voilà!

### Why is it useful?
The numerical benefit is not obvious at first: if one has access to $$A \in \bbR^{d \times d}$$, computing its trace costs $$\mathcal{O}(d)$$, while the above formula requires $$\mathcal{O}(d^2)$$ to compute $$z^\top A z$$, not to mention multiple Monte-Carlo samples for the expectation. But there are cases in which one wants to compute the trace of a matrix, without having explicit access to said matrix!

The flagship example is the full Jacobian of a neural network $$f$$.
Explicitely computing it is out of reach: with backpropagation one can only compute Jacobian-vector products, aka $$\jac_f(x) v$$ for $$v \in \bbR^d$$.
To compute the full Jacobian means computing $$\jac_f(x) e_i$$ for all canonical vectors $$e_i$$, hence calling backpropagation $$d$$ times.
But, to compute the *trace* of the Jacobian, one can sample a single $$z$$, and then approximate the expectation in \eqref{eq:hutchinson} by a (single) Monte-Carlo estimate:

$$
\tr J_f(x) \approx z^\top \left(J_f(x) z \right)
$$

where $$J_f(x) z$$ is computed with a single backpropagation pass. This is what is done in Continuous normalizing flows, in the Ordinary Differential Equation required to evaluate the log likelihood loss. Very clever and elegant!


### Is it a good estimator?

The Hutchinson estimator is a very elegant trick with a short proof, but it has a drawback: its variance is terrible.

**Proposition 1**: For standard Gaussian $$z$$ the variance of the estimator $$z^\top A z$$ is $$2 \norm{A}_F^2$$.
<details><summary>Click to expand the proof </summary>
    Wlog assume that \(A\) is symmetric: we can do so because \(A\) has same trace as its symmetric part, and \(z^\top A z = z^\top (A^\top + A) z / 2 \).
    We can use that a Gaussian \(z \) is still Gaussian under orthogonal transform to assume \(A\) is diagonal.
    Then for $A = \Lambda = \diag(\lambda_1, \ldots, \lambda_d)$,
    \begin{align}
        \bbE[(z^\top \Lambda z)^2]
        &= \bbE [(\sum_i \lambda_i z_i^2)^2] \\
        &= \bbE [(\sum_i \lambda_i^2  z_i^4 + \sum_{i \neq j} \lambda_i \lambda_j z_i^2 z_j^2)] \\
        &= 3\sum_i \lambda_i^2  + \sum_{i \neq j} \lambda_i \lambda_j  \\
        &= 2\sum_i \lambda_i^2  + \sum_i \lambda_i^2 + \sum_{i \neq j} \lambda_i \lambda_j  \\
        &= 2\sum_i \lambda_i^2  + \sum_{i, j} \lambda_i \lambda_j  \\
        &= 2 \norm{A}^2_F + (\tr A)^2
    \end{align}
    using that expectation of $z_i^4 = 3 \sigma^2 = 3$, and $z_i^2$ and $z_j^2$ are independent for $i \neq j$ (with expectation equal to 1)

    Since $\bbE[z^\top A z] = \tr A$, the final variance is $2 \norm{A}^2_F$.
</details>
<br>
This result is quite disappointing: the variance scales like $$\mathcal{O}(d^2)$$, while the trace scales like $$d$$. Can we improve this with other random variables? Yes, but not by much!

**Proposition 2**: For uniform sign $$z$$ (aka Rademacher variables, taking values +1 or -1 with probability 1/2), the variance of the estimator is $2 \sum_{i \neq j} A_{ij}^2$.
It is the law of $$z$$ that minimizes the variance of the Hutchinson estimator.


<details><summary> Click to expand the proof </summary>

Again assume that $A$ is symmetric wlog. Since $z_i^2 = 1$,
\begin{align}
    z^\top A z - \bbE[z^\top A z]
    &= \sum_{i \neq j} A_{ij} z_i z_j \\
    &= 2 \sum_{i < j} A_{ij} z_i z_j
\end{align}
The variables $z_i z_j$ $(i < j)$ are independent; they are also independent Rademacher variables, and so $\Var(z_i z_j) = 1$.
The variance to compute is thus
\begin{align}
    \Var \left(2 \sum_{i < j} A_{ij} z_i z_j\right) = 4 \sum_{i < j} A_{ij}^2 \Var(z_i z_j) = 2 \sum_{i \neq j} A_{ij}^2 \enspace.
\end{align}
For the minimal variance result: for any $z$ (not necessarily Rademacher)
\begin{align}
    \bbE[(z^\top A z)^2]
        &= \bbE[\sum_{ijkl}z_i z_j z_k z_l A_{ij} A_{kl}]
        % &= \bbE[\sum_i A_{ii}^2 z_i^4 + \sum_{i, j} z_i^2 z_j^2 (A_{ii}^2 + 2 A_{ij}^2)]
\end{align}
Now the expectation is 0 as soon as one index is not equal to any of the others ($\bbE[z_i] = 0$ + independence).
So we have four cases to consider:
<ul>
<li> $i=j=k=l$, yielding $\sum_i z_i^4A_{ii}^2$ </li>
<li> $i=j$, $k=l$, $i \neq k$, yielding $\sum_{i \neq k} z_i^2 z_k^2 A_{ii} A_{kk}$ </li>
<li> $i=k$, $j=l$, $i \neq j$, yielding $\sum_{i \neq j} z_i^2 z_j^2 A_{ij}^2$</li>
<li> $i=l$, $j=k$, $i \neq j$, yielding $\sum_{i \neq j} z_i^2 z_j^2 A_{ij}^2$ too</li>
</ul>

Hence by renaming $k$ into $j$ in the second case
\begin{align}
    \bbE[(z^\top A z)^2]
        &= \bbE[\sum_i A_{ii}^2 z_i^4 + \sum_{i \neq j} z_i^2 z_j^2 (A_{ii} A_{jj} + 2 A_{ij}^2)] \\
        &= \sum_i A_{ii}^2 \bbE[z_i^4] + \sum_{i \neq j} (A_{ii} A_{jj} + 2 A_{ij}^2)
\end{align}
since $\bbE[z_i^2 z_j^2] = \bbE[z_i^2] \bbE[z_j^2] = 1$ when $i \neq j$.
To minimize $\bbE[(z^\top A z)^2]$ we should thus minimize $\bbE[z_i^4]$, which by Jensen inequality is lower bounded by $\bbE[z_i^2]^2 = 1$.
The value 1 is achieved for $z_i$ being Rademacher.

</details>

<br>
In hindsight, this optimality result makes sense: if $$A$$ is diagonal, then for any draw of a Rademacher $$z$$, $$z^\top A z = \tr A$$ !
With a small numerical experiment (entries of $$A \in \bbR^{20 \times 20}$$ i.i.d. standard Gaussian) we can check that Rademacher gives a slightly smaller variance.
<div style="text-align: center;">
<img src="/assets/img/hutchinson.svg" style="width:100%; max-width:600px;">
</div>

If we increase the diagonal entries of $$A$$ (by 5 here), we see the variance for Gaussian $$z$$ increase as predicted by Proposition 1, while the performance for Rademacher $$z$$ is unaffected (coherent with Proposition 2)
<div style="text-align: center;">
<img src="/assets/img/hutchinson2.svg" style="width:100%; max-width:600px;">
</div>


<br>
### Beyond trace: computing the full diagonal

As my colleague [Antoine Gonon](https://agonon.github.io/) pointed to me, the Hutchinson trick can be used to compute all the entries of the diagonal:

$$
\bbE_z [(Az) \odot z] = \diag(A)
$$

The proof is very similar to the first one. This technique allows evaluating the diagonal of the Hessian of a neural network $$g: \bbR \to \bbR$$, using Hessian-vector product (see [this great blog post](https://iclr-blogposts.github.io/2024/blog/bench-hvp/)), which has applications in neural network pruning for example.
