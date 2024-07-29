---
layout: post
title:  "Finding equivalent for sequences: the alpha trick"
date:   2024-07-22 00:00:00 +0200
permalink: /:title/
---

The alpha trick is a very cute and efficient trick to find equivalent to sequences defined recursively as $$x_{k+1} = f(x_k)$$.

When trying to find an equivalent to the sequence defined by $$x_{k+1} = f(x_k)$$, proceed as follows:
- first show that $$x_k$$ converges, and find a transformation of $$x_k$$ into $$u_k$$ (e.g. $$u_k = 1 / x_k$$, etc) such that $$u_k \to 0$$
- then do a limited development of the recursive equation satisfied by $$u_k$$ around 0
- **the key part**: raise the limited development equation to a power $$\alpha$$ such that $$u_{k+1}^\alpha - u_k^\alpha$$ converges to a non zero limit, say $$\ell$$:

$$u^\alpha_{k + 1} - u^\alpha_{k} = \ell + o(1) $$

- apply Cesaro to obtain that $$\frac{\sum_{k=1}^n u^\alpha_{k + 1} - u^\alpha_{k}}{n} \to \ell$$, which means that $$u_{k}^\alpha \sim \ell k$$.
- convert this back to an equivalent in $$x_k$$

### First example: gradient descent on the exponential
The best application I know of this trick is to study the iterates of gradient descent on $$F: x \mapsto \exp(-x)$$.
My motivation to to study this was that a minimizer does not exist ($$F > 0$$ but $$F$$ goes to 0 at infinity), so the classical convergence rates results in $$\mathcal{O}(1/k)$$ do not apply.
I wondered if we could say anything about the convergence speed anyways?

Consider gradient descent iterates on $$f$$, started at 0.
We easily see that such iterates will remain positive; when restricted to $$\mathbb{R}_+$$, $$f$$ is 1-smooth, so we take a stepsize of 1.
Our iterates are thus:

$$x_{k+1} = x_k - \exp(-x_k) \, .$$

Writing $$u_k = f(x_k) = \exp(-x_k)$$, one obtains from the above equation:
$$u_{k+1} = u_k \exp(-u_k)$$, with $$u_0 = 1 > 0$$.

The sequence $$(u_k)$$ remains positive, hence $$\exp(-u_k) \leq 1$$ and $$u_k$$ decreases.
Being in addition lower bounded, $$u_k$$ converges; the limit must be a fixed point: it is 0.
Hence, by doing a limited development of $$u_{k} \exp(-u_k)$$ around 0:

$$
\begin{align}
    u_{k+1} &= u_k (1 - u_k + o(u_k)) \\
    u_{k+1}^\alpha &= u_k^\alpha (1 - u_k + o(u_k))^\alpha \\
        &= u_k^\alpha (1 - \alpha u_k + o(u_k))\\
        &= u_k^\alpha - \alpha u_k^{\alpha + 1} + o(u_k^{\alpha + 1})
\end{align}
$$
We see the power of the trick here: by raising the equation to the power $$\alpha$$, we will be able to transform the $$-\alpha u_k^{\alpha + 1}$$ into a constant!
Indeed, picking $$\alpha=-1$$ allows us to obtain $$u_{k+1}^{-1} - u_k^{-1} \to 1$$.
Hence $$\frac{\sum_0^n u_{k+1}^{-1} - u_k^{-1}}{n+1} \to 1$$, and $$u_k \sim 1/k$$.

So we retain the convergence rate in $$1/k$$, which I find very nice!

One may wonder at which speed $$x_k$$ goes to infinity; here $$u_k \to 0$$ so it stays bounded away from 1 at infinity, and thus we can compose the equivalent by the log function: $$\log u_k \sim \log(1/k)$$, and since $$\log u_k = -x_k$$, the GD iterates $$x_k$$ are equivalent to $$\log k$$.

## Second example: Nesterov's sequence
Let $$t_{k+1} = \frac{1 + \sqrt{1 + 4 t_k^2}}{2}$$, the infamous sequence appearing in Nesterov's original accelerated algorithm.
Let's use the alpha trick to find an equivalent to $$t_k$$.
One has $$t_{k+1} \geq t_k +1/2$$ so we cannot hope for convergence; consider $$u_k = 1/t_k$$ instead:

$$
u_{k+1} = \frac{2}{1 + \sqrt{1 + 4 / u_k^2}}
$$

<!-- This sequence is positive and decreasing, so it converges, and its limit is a fixed point, so it's 0. -->
SInce $$t_k \to + \infty$$, $$u_k \to 0$$, and

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


<!-- This shows why people often study the momentum in Nesterov/FISTA as $$\frac{k - 1}{k + 2}$$.  -->
