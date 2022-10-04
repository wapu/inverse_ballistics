import os
import warnings
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

from sklearn.cluster import MeanShift
from sklearn.neighbors.kde import KernelDensity


latex_fonts = {
    'mathtext.fontset': 'cm', # or 'stix'
    'font.family': 'cmss10', # or 'STIXGeneral
    "text.usetex": True,
    "axes.labelsize": 10,
    "font.size": 16,
    "legend.fontsize": 10,
}
mpl.rcParams.update(latex_fonts)



class InverseBallisticsModel():

    n_parameters = 4
    n_observations = 1
    name = 'inverse-ballistics'

    def __init__(self, g=9.81, k=0.25, m=0.2):
        self.name = 'inverse-ballistics'

        self.g = g # gravity
        self.k = k # drag coefficient dependent on object shape and traversed medium
        self.m = m # object mass

        self.xy_mu = np.array((0, 1.5))
        self.xy_std = np.array((0.5, 0.5))

    def sample_prior(self, N):
        x = np.random.randn(N, 1) * self.xy_std[0] + self.xy_mu[0]
        y = np.random.randn(N, 1) * self.xy_std[1] + self.xy_mu[1]
        y = np.maximum(y, 0)
        angle = np.random.rand(N, 1) * np.pi/2 * 0.8 + np.pi/2 * 0.1
        v0 = np.random.poisson(15, (N, 1))
        return np.concatenate([x, y, angle, v0], axis=1)

    def trajectories_from_parameters(self, x):
        x0, y0, angle, v0 = np.split(x, 4, axis=1)
        v0 = np.repeat(v0, 1500, axis=-1)
        angle = np.repeat(angle, 1500, axis=-1)
        t = np.repeat(np.linspace(0, 6, 1500)[None,:], x.shape[0], axis=0)
        vx = v0 * np.cos(angle)
        vy = v0 * np.sin(angle)

        expterm = np.exp(-self.k*t / self.m) - 1
        xt = x0 - (vx*self.m / self.k) * expterm
        yt = y0 - (self.m/(self.k*self.k)) * ((self.g*self.m + vy*self.k) * expterm + self.g*t*self.k)
        return xt, yt

    def impact_from_trajectories(self, xs, ys):
        ys_peak = np.argmax(ys, axis=1)
        ys_after_peak = np.where(xs < xs[np.arange(xs.shape[0]), ys_peak][:,None], 0.1, ys)
        xs_impact = xs[np.diff(np.signbit(ys_after_peak)).nonzero()]
        return xs_impact

    def forward_process(self, x):
        xs, ys = self.trajectories_from_parameters(x)
        return self.impact_from_trajectories(xs, ys)[:,None]

    def init_plot(self, y_target):
        return plt.figure(figsize=(8,8))

    def update_plot(self, x, y_target):
        plt.gcf().clear()
        x = np.array(x)
        xs, ys = self.trajectories_from_parameters(x)

        # Trajectories
        lines = np.stack([xs, ys], axis=-1)
        lines = [np.squeeze(line) for line in np.split(lines, len(lines))]
        line_collection = LineCollection(lines, linewidths=1, alpha=0.1, rasterized=True)
        plt.gca().add_collection(line_collection)

        # Arrows for initial velocity
        x0, y0, angle, v0 = [np.squeeze(p) for p in np.split(x, 4, axis=1)]
        vx = v0 * np.cos(angle)
        vy = v0 * np.sin(angle)
        plt.quiver(x0, y0, vx, vy, angles='xy', scale_units='xy',
                   scale=15, width=0.001, headwidth=3, color='red', alpha=0.2,
                   zorder=10, rasterized=True)

        # Impact points
        xs_impact = self.impact_from_trajectories(xs, ys)
        plt.scatter(xs_impact, np.zeros(xs_impact.shape),
                    s=5, edgecolor='red', facecolor='white', alpha=0.2,
                    zorder=20, rasterized=True)

        # Target
        plt.axhline(0, color='k', linestyle='dotted', linewidth=1)
        plt.axvline(y_target, color='k', linewidth=1)

        plt.gca().set_aspect('equal', 'datalim')
        plt.xlim([np.amin(xs) - 0.5, np.amax(xs_impact) + 0.5])
        plt.ylim([-0.5, np.amax(ys) + 0.5])
        plt.gca().set_xticks([]); plt.gca().set_yticks([])
        plt.tight_layout(pad=0, w_pad=-0.5, h_pad=-0.5)

    def find_MAP(self, x):
        try:
            mean_shift = MeanShift()
            mean_shift.fit(x)
            centers = mean_shift.cluster_centers_
            kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(x)

            best_center = (None, -np.inf)
            dens = kde.score_samples(centers)
            for c,d in zip(centers, dens):
                if d > best_center[1]:
                    best_center = (c.copy(), d)

            dist_to_best = np.sum((x - best_center[0])**2, axis=1)
            return np.argmin(dist_to_best)
        except:
            print('Mean shift failed')
            return 0

    def arcarrow(self, start, direction, dist=2, open_angle=45,
                 kw=dict(arrowstyle='<->, head_width=2, head_length=2', ec='black', lw=1)):
        angle = np.arctan2(direction[1], direction[0])
        angle1 = angle - np.radians(open_angle/2)
        x1 = start[0] + dist * np.cos(angle1)
        y1 = start[1] + dist * np.sin(angle1)
        angle2 = angle + np.radians(open_angle/2)
        x2 = start[0] + dist * np.cos(angle2)
        y2 = start[1] + dist * np.sin(angle2)
        plt.gca().add_patch(patches.FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f"arc3, rad=.6", **kw))
        plt.text(x1+0.6, y1, r'$x_3$', ha='center', va='center')

    def plot_sample(self, x, xs=None, ys=None, colors={}, alphas={}, annotate=False, y_target=None, xlim=[-2, 18], ylim=[-1.5, 6.5]):
        c = {'lines': (.5,.5,.5), 'arrows': (.2,.2,.2), 'impact': '#96BF0D'}
        c.update(colors)
        colors = c
        a = {'lines': 0.015, 'arrows': 0.4, 'impact': 0.2, 'density': 0.2}
        a.update(alphas)
        alphas = a

        if xs is None or ys is None:
            xs, ys = self.trajectories_from_parameters(x)
        exemplar = self.find_MAP(x)

        # Trajectories
        lines = np.stack([xs, ys], axis=-1)
        lines = [np.squeeze(line) for line in np.split(lines, len(lines))]
        line_collection = LineCollection(lines, colors=colors['lines'], linewidths=1, alpha=alphas['lines'], zorder=1, rasterized=True)
        plt.gca().add_collection(line_collection)
        # Exemplar trajectory
        plt.plot(xs[exemplar], ys[exemplar], color=(0,0,0), linewidth=1, linestyle='dashed', zorder=100)

        # Arrows for initial velocity
        x0, y0, angle, v0 = [np.squeeze(p) for p in np.split(x[:150], 4, axis=1)]
        vx = v0 * np.cos(angle)
        vy = v0 * np.sin(angle)
        plt.quiver(x0, y0, vx, vy, angles='xy', scale_units='xy',
                   scale=15, width=0.001, headwidth=7, color=colors['arrows'], alpha=alphas['arrows'],
                   zorder=10, rasterized=True)
        # Exemplar arrow
        x0, y0, angle, v0 = x[exemplar]
        vx = v0 * np.cos(angle)
        vy = v0 * np.sin(angle)
        plt.arrow(x0, y0, vx/5, vy/5,
                  width=0.003, head_width=0.15, color=(0,0,0),
                  zorder=101)
        plt.scatter([x0-0.02], [y0], s=10, edgecolor='black', facecolor='white', zorder=102)
        if annotate:
            self.arcarrow([x0,y0], [vx/5,vy/5])
            plt.text(x0+vx/5 - 0.1, y0+vy/5 + 0.7, r'$x_4$', ha='center', va='center')
            plt.text(-0.4, 0.8, r'$(x_1, x_2)$', ha='center', va='center')

        # Impact points
        xs_impact = self.impact_from_trajectories(xs, ys)
        exemplar_impact = self.impact_from_trajectories(xs[exemplar:exemplar+1], ys[exemplar:exemplar+1])
        if len(xs_impact) > 0:
            plt.scatter(xs_impact, np.zeros(xs_impact.shape),
                        s=5, edgecolor=colors['impact'], facecolor='white', alpha=alphas['impact'],
                        zorder=20, rasterized=True)
            if len(exemplar_impact) > 0:
                plt.scatter([exemplar_impact], [0], s=10, edgecolor='black', facecolor='white',
                            zorder=102)
                if annotate:
                    plt.text(exemplar_impact-0.4, -0.7, r'$y$', ha='center', va='center')
            # Density
            from scipy.stats import gaussian_kde
            density = gaussian_kde(xs_impact)
            density.covariance_factor = lambda: .15
            density._compute_covariance()
            domain = np.linspace(np.amin(xs_impact)-.5, np.amax(xs_impact)+.5, 200)
            density = density(domain)
            plt.fill_between(domain, 3*density/np.amax(density), color=colors['impact'], alpha=alphas['density'])

        # X axis
        plt.axhline(0, color='gray', linestyle='dotted', linewidth=1, alpha=.5)
        if y_target is not None:
            plt.axvline(y_target, color='gray', linestyle='dotted', linewidth=1, alpha=.5)

        plt.gca().set_aspect('equal', 'datalim')
        plt.gca().set_xticks([]); plt.gca().set_yticks([])
        plt.xlim(xlim); plt.ylim(ylim)



class InverseBallisticsDataset(Dataset):

    def __init__(self, model, n, root_dir=None, suffix=''):
        self.model = model
        self.root_dir = root_dir
        if root_dir is None:
            warnings.warn('InverseBallisticsDataset: No data directory specified, generated data will not be stored.', Warning)
        self.n = n
        self.suffix = suffix
        if len(suffix) > 0 and not '_' in suffix[:1]:
            suffix = '_' + suffix

        try:
            x = np.load(f'{root_dir}/{self.model.name}_x{suffix}.npy')[:n,...]
        except Exception as e:
            print(f'InverseBallisticsDataset: Not enough data for model "{self.model.name}" found, generating {n} new samples...')
            x = model.sample_prior(n)
            if root_dir is not None:
                os.makedirs(root_dir, exist_ok=True)
                np.save(f'{root_dir}/{self.model.name}_x{suffix}', x)
        self.x = x
        try:
            y = np.load(f'{root_dir}/{self.model.name}_y{suffix}.npy')[:n,...]
        except Exception as e:
            print(f'InverseBallisticsDataset: Not enough labels for model "{self.model.name}" found, running forward process on {n} samples...')
            y = []
            if n > 100000:
                for i in range((n-1)//100000 + 1):
                    print(f'InverseBallisticsDataset: Forward process chunk {i+1}...')
                    y.append(model.forward_process(x[100000*i : min(n, 100000*(i+1)),...]))
                y = np.concatenate(y, axis=0)
            else:
                y = model.forward_process(x)
            print()
            if root_dir is not None:
                np.save(f'{root_dir}/{self.model.name}_y{suffix}', y)
        self.y = y

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.item()
        return self.x[i], self.y[i]

    def get_dataloader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=True, drop_last=True)




if __name__ == '__main__':
    pass

    model = InverseBallisticsModel()
    # train_data = InverseBallisticsDataset(model, 10000, 'bal_data', suffix='train')
    # train_loader = train_data.get_dataloader(1000)

    plt.figure(figsize=(10,4))
    np.random.seed(0)
    model.plot_sample(model.sample_prior(2000), annotate=True, y_target=0)
    plt.gcf().tight_layout()
    plt.show()
