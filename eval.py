import numpy as np
import torch
import time
import pickle

from data import InverseBallisticsModel
from metrics import quantile_ABC, multi_mmd, resimulation_error



# Draw N samples from the trained conditional density model
# for conditional input (observation) y_target,
# to be used once per observation during evaluation
def sample_evaluated_model(model, y_target, N):

    # YOUR
    # CODE
    # HERE

    pass



# Generate a very large pool of paired data
# for the Approximate Bayesian Computation baseline
def prepare_ABC_samples(N=int(1e8), data_model=InverseBallisticsModel(), sample_path='abc'):
    print(f'Drawing {N:,} samples from prior...', end=' ')
    t = time.time()
    x = data_model.sample_prior(N).astype(np.float32)
    y = data_model.forward_process(x).astype(np.float32)
    np.save(f'{sample_path}/x_huge', x)
    np.save(f'{sample_path}/y_huge', y)
    print(f'Done in {time.time()-t:.1f} seconds.')



# Draw n_runs samples from the marginal distribution of y_target,
# approximate the ground truth posteriors with quantile ABC
# and evaluate the provided conditional density model against them
def evaluate(n_runs, model, data_model=InverseBallisticsModel(), sample_path='abc'):
    try:
        x = np.load(f'{sample_path}/x_huge.npy')
        y = np.load(f'{sample_path}/y_huge.npy')
    except:
        print('No paired data for ABC preparation found in "{sample_path}"\n')

    posterior_mismatches = []
    times = []
    resim_errors = []

    for i in range(n_runs):
        print(f'Run {i+1:06}/{n_runs:06}:')

        # Get randomized y_target and its approximate ground truth posterior
        try:
            with open(f'{sample_path}/{i:05}.pkl', 'rb') as f:
                (y_target, gt_sample, threshold) = pickle.load(f)
        except:
            print(f'No ABC preparation found in "{sample_path}", creating new y_target')
            y_target = data_model.forward_process(data_model.sample_prior(1)).astype(np.float32)
            try:
                gt_sample, threshold = quantile_ABC(x, y, y_target)
            except:
                print('Needs paired data, please call "prepare_ABC_samples()" first!')
                return
            with open(f'{sample_path}/{i:05}.pkl', 'wb') as f:
                pickle.dump((y_target, gt_sample, threshold), f)

        print(f'y_target = {np.round(y_target[0], 3)})')

        # Evaluate density model for this posterior
        with torch.no_grad():
            t_start = time.time()
            sample = sample_evaluated_model(model, y_target, N=500)
            t_end = time.time()
            gt_sample = torch.from_numpy(gt_sample).to(sample.device)

            posterior_mismatch = multi_mmd(sample, gt_sample).item()
            posterior_mismatches.append(posterior_mismatch)
            resim_error = resimulation_error(y_target, sample).item()
            resim_errors.append(resim_error)
            t = t_end - t_start
            times.append(t)

        print(f'posterior mismatch {posterior_mismatch:.5f} | resimulation error {resim_error:.5f} | time {t:.3f}s \n')

    # Results
    print(f'Mean over {n_runs} runs:')
        print(f'posterior mismatch {np.mean(posterior_mismatches):.5f} | resimulation error {np.mean(np.minimum(100, resim_errors)):.5f} | time {np.mean(times):.3f}s')




if __name__ == '__main__':
    pass

    # prepare_ABC_samples()
    # evaluate(n_runs=1000)
