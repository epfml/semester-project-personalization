The paper I sent you can be a baseline (FedAvg). The Scaffold part can be ignored as we are not really going for the direction.

Their FedAvg-P under a special condition where
- use gradient instead of momentum, i.e. $\eta_u=\eta_v=1$. The beginning of section 2.1 tells us that in this case they choose step size $\gamma_u=\gamma_v$
- all clients participate in the federated learning, i.e. m=n
- fix 1 local step, i.e. K=1

Our algorithm under a special condition where
- there are n groups, one client in each group.

In this case, the only difference between our algorithm is that the learning rate for private layers. We scale the learning rate with 1/n whereas they don't. Our methods may have at least two advantages:
- our algorithm is naturally derived from GD whereas theirs is not.
- their convergence rate in this situation (checkout Table 2) does not enjoy $1/n$ scaling because of the variance of $\sigma_v^2$. I think when we scale down the learning rate of private layers, we can get $1/n$ scaling based on the sketch proof we have from the last discussion.

Could you double check that the our analysis enjoy the 1/n scaling? If so, we already have something valuable to say in the paper. Please also check what are the empirical results between two methods.
