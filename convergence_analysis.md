We can start with a simplest case. Suppose that we already know that n workers can be classified into two groups ($G_1, G_2$). In this case we don't need the 3rd condition (in the last image I sent) to capture the dissimilarity between two groups and we don't need 1st condition to capture the similarity between workers in the same group. We will include these conditions in the future. 

In the first group, they use model weight $[x_s; x_d]$ while in the second group, they use $[x_s; x_{d}']$. For client $i\in G_1$, its objective is $f_i(x_s, x_d)$. Let's denote $\bar{f}^1(x_s, x_d)=\frac{1}{|G_1|} \sum_{i\in G_1} f_i(x_s, x_d)$. Similarly we can define $\bar{f}^2(x_s, x_d')$. The global objective is that

$$
f(x_s, x_d, x_d')=\frac{1}{n} (\sum_{i\in G_1} f_i(x_s, x_d) + \sum_{i\in G_2} f_i(x_s, x_d'))
$$

Now the problem becomes an optimization problem with an "augmented" model $x_s, x_d, x_d'$. We can compute its full gradient $\nabla f(x_s, x_d, x_d')=[\partial_{x_s} f(x_s, x_d, x_d'); \partial_{x_d} f(x_s, x_d, x_d'); \partial_{x_d'} f(x_s, x_d, x_d')]$ where

$$
\partial_{x_s} f(x_s, x_d, x_d') = \frac{|G_1|}{n} \partial_{x_s} \bar{f}^1(x_s, x_d) + \frac{|G_2|}{n} \partial_{x_s} \bar{f}^2(x_s, x_d')
$$

$$
\partial_{x_d} f(x_s, x_d, x_d') = \frac{|G_1|}{n} \partial_{x_d} \bar{f}^1(x_s, x_d)
$$

$$
\partial_{x_d'} f(x_s, x_d, x_d') = \frac{|G_2|}{n} \partial_{x_d'} \bar{f}^2(x_s, x_d')
$$

Now we can construct a simple stochastic estimate of the above full gradient simply by computing the stochastic estimate of $\nabla \bar{f}^1(x_s, x_d)$ and $\nabla \bar{f}^2(x_s, x_d')$ and rearrage the coordinates and multiply constants in a similar manner. 
​
If we assume the objectives are smooth and non-convex, we can already derive the convergence rate of this stochastic estimate. The proof is almost identical to the proof of SGD for $\min_x f(x)$. The only difference is that now our stochastic estimate has a different variance $\sigma^2$. Let us assume the variance of $\partial_{x_s} f_i(x_s, x_d)$ is $\sigma_{s}^2$ and $\tilde{\sigma}_{s}^2$ for $G_1$ and $G_2$ respectively and the variance of $\partial_{x_d} f_i(x_s, x_d)$ and $\partial_{x_{d'}} f_i(x_s, x_d)$ are $\sigma_d^2$ and $\tilde{\sigma}_d^2$. Note that we assume all randomness are independent and therefore the variance of stochastic estimates of $\nabla\bar{f}^1$, $\nabla\bar{f}^2$ reduce with the number of gradients they averaged. Then

$$
\sigma^2 = \left(\frac{\frac{|G_1|}{n}\sigma^2_s+\frac{|G_2|}{n}\tilde{\sigma}^2_s}{n} \right) + \left(\frac{|G_1|}{n^2}\sigma_d^2\right)+ \left(\frac{|G_2|}{n^2}\sigma_{d'}^2\right)
$$

and we can plug this into the SGD proof and analyze if the variance scales with $n$.
​
**However**, it is crucial that the above stationary point may not be the stationary point of $\bar{f}^1(x_s, x_d)$ or $\bar{f}^2(x_s, x_d')$! The above stationary point only ensures

$$
\frac{|G_1|}{n} \partial_{x_s} \bar{f}^1(x_s, x_d) + \frac{|G_2|}{n} \partial_{x_s} \bar{f}^2(x_s, x_d')=0
$$

$$
 \frac{|G_1|}{n} \partial_{x_d} \bar{f}^1(x_s, x_d) = 0
$$

$$
 \frac{|G_2|}{n} \partial_{x_{d}'} \bar{f}^2(x_s, x_d')=0
$$

It means $\bar{f}^1(x_s, x_d)$ may not be the best. Therefore we need an additional assumption to ensure that both groups can reach their own stationary points as well. We may propose the following "generalized strong growth condition"

$$
\lVert \partial_{x_s} \bar{f}^1(x_s, x_d) - \partial_{x_s} \bar{f}^2(x_s, x_d') \rVert_2^2 \le M \lVert \nabla f(x_s, x_d, x_d') \rVert_2^2
$$

Then we can ensure the desired property.
​
This simplified problem can be generalized/explored in many ways to make it more interesting
- $x_d$ and $x_{d'}$ does not have to have same shape;
- generalize to more groups;
- we don't know the groups in advance (require more assumptions);
- alternative algorithms;
- alternative strong growth conditions.
- while the above algorithm we use $x_s$ and $x_d$ to denote shallow layers and deep layers, the proof seems to still work if we use $x_s$ as any shared layers and $x_d$ as the rest of layers? Need to check if this is the case.
