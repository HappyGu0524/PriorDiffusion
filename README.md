# PriorDiffusion
Discrete Modeling via Boundary Conditional Diffusion Processes. *NeuraIPS 2024*

## Quick Overview
In an effort to improve the generalization capabilities for discrete modeling tasks, we endeavor to incorporate **the discreteness as an a priori constraint within the trajectory modeling paradigm** of continuous diffusion processes.

Our framework is a module constructed on current diffusion models, where the pseudo code of kernel part ***rescale diffusion trajectory*** is:
```Python traceback
def rescale_diffusion_trajectory(x_0, epsilon, embedding, 
        labels, alphas_cumprod, timesteps, mode):
    #embedding: embedding matrix, f(x,i)=(embedding * x)[i]
    #labels: I
    #alphas_cumprod: list of all u_t
    #timesteps: t
    #mode: noising or denoising

    #1. get f(x,i):
    self_dot = torch.sum(embedding * embedding, dim=-1)
    f_x_i = self_dot[labels][..., None]
    labels = labels[..., None]

    #2. get f(x,j) and f(eps,j):
    embedding = embedding.permute(1, 0)
    f_x_j = torch.matmul(x_0, embedding)
    f_eps_j = torch.matmul(epsilon, embedding)

    #3. get f(x,i) - f(x,j): (usually >=0; smaller -> closer)
    #filter out f(x,i)-f(x,i) with a large positive number 100
    fxi_minus_fxj = (f_x_i - f_x_j).scatter(-1, labels, 100)

    #4. get f(eps,i) and f(eps,j) - f(eps,i): (larger -> more noise)
    f_eps_i = torch.gather(f_eps_j, -1, labels)
    #filter out f(eps,i)-f(eps,i) with a large negative number -100
    fepsj_minus_fepsi = (f_eps_j - f_eps_i).scatter(-1, labels, -100)

    #5. get fraction and u_t_0
    #mask results outside the support set
    info_mask = (fepsj_minus_fepsi < 0) | (fxi_minus_fxj < 0)
    fraction = fix_minus_fjx / fjeps_minus_fieps
    fraction[info_mask] = 100
    min_frac, _ = fraction.min(dim=-1) # minimum
    #Diffusion Variance Preserving eq. (9)
    u_t_0 = torch.sqrt(1 / (1 + min_frac ** 2))[..., None]

    #6. rescale timesteps
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    
    ###!!!important trick!!!###
    #We do not need to calculate the function G(x_0,t) (eq. (12)).
    #Timesteps of diffusion processes are discrete and
    #  we just iterate over and compare with all coefficient functions.
    #Besides, function G is easy to calculate for Flow Matching.
    index = torch.sum(u_t_0 < sqrt_alphas_cumprod, dim=-1)

    #T is the maximum timestep, for example T=2000.
    #confactor is the confidency factor
    #tau is the rescaled timestep
    #delta_tau is the rescaled decoding velocity
    if mode == 'noising':
        tau = (timesteps + index - \
            (((timesteps + 1) / T) * index)).long().clamp(0, T)
        tau = (confactor * tau.float() + \
            (1.0 - confactor) * timesteps.float()).long().clamp(0, T)
        return tau
    elif mode == 'denoising':
        delta_tau = (T - index) / T
        delta_tau = (confactor * delta_tau + \
            (1 - confactor) * 1.0).clamp(0, 1)
        return delta_tau
```
