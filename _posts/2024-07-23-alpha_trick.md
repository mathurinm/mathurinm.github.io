---
layout: post
title:  "Finding equivalent for sequences: the alpha trick"
date:   2024-07-22 00:00:00 +0200
permalink: /:title/
---

A very cute and efficient trick to find equivalent to sequences defined recursively as $$x_{k+1} = f(x_k)$$.

When trying to find an equivalent to the sequence defined by $$u_{k+1} = f(u_k)$$
-  first show that $$u_k$$ converges (say to 0)
- then do a limited development
- raise it to a power $$\alpha$$ such that $$u_{k+1}^\alpha - u_k^\alpha$$ converges to a non zero limit, say $$\ell$$
- apply Cesaro to obtain that $$u_{k+1}^\alpha \sim \ell k$$.


### First example: gradient descent on the exponential
The iterates of gradient descent with stepsize 1 on $$\mapsto \exp(-x)$$, started at 0, are:

$$x_{k+1} = x_k - \exp(-x_k) \, .$$

Writing $$u_k = \exp(-x_k)$$, one obtains:
$$u_{k+1} = u_k \exp(-u_k)$$, with $$u_0 > 0$$.

The sequence $$(u_k)$$ remains positive, hence $$\exp(-u_k) \leq 1$$ and $$u_k$$ decreases.
Being in addition lower bounded, $$u_k$$ converges; the limit must be a fixed point: it is 0.
Hence

$$
\begin{align}
    u_{k+1} &= u_k (1 - u_k + o(u_k)) \\
    u_{k+1}^\alpha &= u_k^\alpha (1 - u_k + o(u_k))^\alpha \\
        &= u_k^\alpha (1 - \alpha u_k + o(u_k))\\
        &= u_k^\alpha - \alpha u_k^{\alpha + 1} + o(u_k^{\alpha + 1})
\end{align}
$$

Pick $$\alpha=-1$$ to obtain $$u_{k+1}^{-1} - u_k^{-1} \to 1$$.
Hence $$\frac{\sum_0^n u_{k+1}^{-1} - u_k^{-1}}{n+1} \to 1$$, and $$u_n \sim 1/n$$.


## Second example: Nesterov's sequence
Let $$t_{k+1} = \frac{1 + \sqrt{1 + 4 t_k^2}}{2}$$, the sequence appearing in Nesterov's original accelerated algorithm.
One has $$t_{k+1} \geq t_k +1/2$$ so we cannot hope for convergence; consider $$u_k = 1/t_k$$ instead:

$$
u_{k+1} = \frac{2}{1 + \sqrt{1 + 4 / u_k^2}}
$$

This sequence is positive and decreasing, so it converges, and its limit is a fixed point, so it's 0.

$$
\begin{align}
    u_{k+1} &= \frac{2}{1 + \sqrt{1 + 4/u_k^2}} \\
    &= \frac{2 u_k}{u_k + 2 \sqrt{u_k^2/4 + 1}} \\
    % &= \frac{2 u_k}{u_k + 2 (1 + u_k^2/8 + o(u_k^2))} \\
    &= \frac{u_k}{1 + u_k/2 + o(u_k)} \\
    &= u_k (1 - u_k/2 + o(u_k)) \\
    u_{k+1}^\alpha &= u_k^\alpha (1 - \alpha u_k/2 + o(u_k)) \\
    u_{k+1}^\alpha &= u_k^\alpha - \alpha u_k^{\alpha + 1}/2 + o(u_k^{\alpha + 1})
\end{align}
$$

Hence picking $$\alpha = -1$$, $$u_{k+1}^{-1} - u_k^{-1} \to 1/2$$ and so $$u_k \sim 2/k$$ and finally $$t_k \sim k/2$$.
