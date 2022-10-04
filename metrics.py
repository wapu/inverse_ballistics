import numpy as np
import torch
from scipy.spatial import distance_matrix


# Apply MMD (Gretton et al) to two sample sets x and y,
# averaging over inverse multiquadratic kernels;
# approaches zero for two large samples from the same distribution
def multi_mmd(x, y, widths_exponents=[(0.2, 0.1), (0.2, 0.5), (0.2, 2)]):
    xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = torch.clamp(rx.t() + rx - 2*xx, 0, np.inf)
    dyy = torch.clamp(ry.t() + ry - 2*yy, 0, np.inf)
    dxy = torch.clamp(rx.t() + ry - 2*xy, 0, np.inf)

    XX, YY, XY = (torch.zeros_like(xx), torch.zeros_like(xx), torch.zeros_like(xx))

    for C,a in widths_exponents:
        XX += C**a * ((C + dxx) / a)**-a
        YY += C**a * ((C + dyy) / a)**-a
        XY += C**a * ((C + dxy) / a)**-a

    return torch.mean(XX + YY - 2*XY)


# Sort the given paired data set (x, y) by how near y is to y_target
# and return the n nearest samples x from that ordering
def quantile_ABC(x, y, y_target, n=500):
    print(f'Evaluating ABC to obtain {n:,} samples closest to {y_target[0]} from set of {len(y):,}...', end=' ')
    t = time.time()
    d = distance_matrix(y_target, y)[0]
    sort = np.argsort(d)[1:]
    sample = x[sort][:n]
    threshold = d[sort[n]]
    print(f'Done in {time.time()-t:.1f} seconds, tolerance is {threshold:.3f}.')
    return sample, threshold


# Draw samples (x, y) from the model until n have been found that are
# within threshold distance of y_target (i.e. rejection sampling, very slow)
def threshold_ABC(y_target, threshold=0.01, n=500):
    print(f'Evaluating ABC to obtain {n:,} samples within {threshold} distance of {y_target[0]}...', end=' ')
    t = time.time()
    n_samples = 0
    sample = []
    t_square = threshold*threshold
    while len(sample) < n:
        x = model.sample_prior(1)
        y = model.forward_process(x)
        if np.sum(np.square(y - y_target)) <= t_square:
            sample.append(x)
        n_samples += 1
    print(f'Done in {time.time()-t:.1f} seconds, generated {n_samples:,} samples.')
    return np.concatenate(sample, axis=0).astype(np.float32)


# Apply the forward process to all samples x and average the
# Euclidean distance between the outcomes and y_target
def resimulation_error(y_target, x):
    y = model.forward_process(x.cpu().numpy())
    dists = torch.sum((y - y_target)**2, dim=1).sqrt()
    return dists.mean()
