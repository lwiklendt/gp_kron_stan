import datetime
import difflib
import logging
import os
from pathlib import Path
import pickle
import pprint
import shutil
import sys
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec
from cmdstanpy import CmdStanModel
import numpy as np
import scipy.stats


# TODO do we need too call cmdstanpy.utils.cxx_toolchain_path()?


def negbinom_mu_phi_to_alpha_beta(mu, phi):
    """
    Converts from Stan's alternative paramaterisation (mu, phi) to the (alpha, beta) parameterisation.
    Where mean: mu = alpha/beta, variance: mu + mu**2/phi = alpha * (beta + 1) / beta**2.
    :param mu: parameter 1
    :param phi: parameter 2
    :return: tuple (alpha, beta)
    """
    # Check:
    #   alpha = phi
    #   beta = phi/mu
    #   mean:
    #     mu = alpha / beta
    #        = phi / (phi / mu)
    #        = mu
    #   variance:
    #     mu + mu**2/phi = alpha * (beta + 1) / beta**2
    #                    = phi * (phi/mu + 1) / (phi/mu)**2
    #                    = phi * (phi/mu + 1) * mu**2/phi**2
    #                    = (phi**2/mu + phi) * (mu**2/phi**2)
    #                    = (phi**2/mu)*(mu**2/phi**2) + phi*(mu**2/phi**2)
    #                    = mu + mu**2/phi
    alpha = phi
    beta = phi / mu
    return alpha, beta


def negbinom_mu_phi_to_numpy(mu, phi):
    """
    Converts from Stan's alternative paramaterisation (mu, phi) to the (n, p) parameterisation in
    numpy and scipy, for n successes and probability p of success.
    Where mean = mu, variance = mu + mu**2/phi.
    :param mu: parameter 1
    :param phi: parameter 2
    :return: tuple (n, p)
    """
    # Check:
    #   variance = mu + mu**2/phi
    #   n = mu**2 / (var - mu)
    #     = mu**2 / (mu + mu**2/phi - mu)
    #     = mu**2 / (mu**2/phi)
    #     = phi
    #   p = n / (n + mu)
    #     = phi / (phi + mu)
    n = phi
    p = phi / (phi + mu)
    return n, p


def sample(src_stan_code: str,
           data: dict | None = None,
           output_dirname: str | Path | None = None,
           sample_kwargs: dict | None = None,
           compile_kwargs: dict | None = None,
           force_resample=False,
           **other_sample_kwargs) -> (dict, bool):
    """
    MCMC samples from a CmdStanModel.

    :param src_stan_code: Stan source code as a str.
    :param data: dict mapping from data block variable names to data. If None then attempt to obtain cached
                 samples.
    :param output_dirname: Location to where artefacts will be written.
    :param sample_kwargs: args that can modify the output, such as seed, adapt_delta etc.
        These kwargs will be included in the hash to check whether we can use the cached
        samples, or whether we need to resample the model. For other args that do not
        change the posterior such as `refresh`, supply as a kwarg via **other_stan_kwargs.
    :param compile_kwargs: kwargs passed to CmdStanModel.
        E.g. compile_kwargs = {'cpp_options': {'STAN_THREADS': True, 'STAN_OPENCL': True}}.
    :param force_resample: force a resample even if otherwise it would not be needed
    :param other_sample_kwargs: kwargs to CmdStanModel.sample() that do not change the posterior
        distribution, such as refresh or show_progress. Any changes to these parameters will
        not trigger a resampling.
    :return: dict of variable names mapping to posterior samples, and bool signifying whether resampling took
             place (True) or we could load from pickle (False).
    """

    # Create filenames and dirnames.
    output_dirname = Path(output_dirname)
    dst_stan_filename = output_dirname / 'model.stan'
    samples_filename = output_dirname / 'samples.pkl'
    data_file = output_dirname / 'data.pkl'
    compile_kwargs_filename = output_dirname / 'kwargs_compile.pkl'
    sample_kwargs_filename = output_dirname / 'kwargs_sample.pkl'
    compile_log_filename = output_dirname / 'log_compile.txt'
    sample_log_filename = output_dirname / 'log_sample.txt'
    ranks_plot_filename = output_dirname / 'ranks.png'
    output_dirname.mkdir(parents=True, exist_ok=True)

    # Ensure kwargs are dict in place of None.
    sample_kwargs = {} if sample_kwargs is None else sample_kwargs
    compile_kwargs = {} if compile_kwargs is None else compile_kwargs

    # ======================= #
    # ----- Compilation ----- #
    # ======================= #

    # If compile_kwargs is the same as cached, then we don't force compile.
    compile = 'force'
    if compile_kwargs_filename.exists():
        with open(compile_kwargs_filename, 'rb') as f:
            compile_kwargs_cached = pickle.load(f)
        if compile_kwargs_cached == compile_kwargs:
            # Turn off forced compile, and go back to weaker compile determination based only on source code change.
            compile = True
    if compile == 'force':
        print('cached compile_kwargs missing or different')

    # Setup logging, DEBUG by default and INFO to stdout.
    logger = logging.getLogger('cmdstanpy')
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    # Diff with existing code to see what's new.
    if dst_stan_filename.exists():
        with open(dst_stan_filename) as f:
            existing_code = f.read()

        if existing_code != src_stan_code:

            if existing_code is not None:
                print('Stan model code is different to the existing version on file (- file, + curr):')
                # print the diff between cached and current code
                result = difflib.unified_diff(existing_code.splitlines(), src_stan_code.splitlines(), n=0, lineterm='')
                print('\n'.join(list(result)[2:]))  # [2:] is to remove the control lines '---' and '+++'
                print('recompiling...')

            with open(dst_stan_filename, 'w') as f:
                f.write(src_stan_code)
    else:
        with open(dst_stan_filename, 'w') as f:
            f.write(src_stan_code)

    # Prepare and add compile log file handler.
    # We don't know whether CmdStanModel will need to compile a new file or not, so we store the log
    # in a temporary file and if a recompilation was indeed performed we copy over that file to
    # compile_log_file.
    with tempfile.TemporaryDirectory() as tempdir:
        compile_log_tempfilename = Path(tempdir) / 'log_compile.txt'
        compile_log_handler = logging.FileHandler(filename=compile_log_tempfilename)
        compile_log_handler.setLevel(logging.DEBUG)
        logger.addHandler(compile_log_handler)

        # Create model, possibly recompiling.
        pre_compile_time = datetime.datetime.now().timestamp()
        m = CmdStanModel(stan_file=dst_stan_filename, compile=compile, **compile_kwargs)
        exe_file = Path(m.exe_file)
        new_exe = False
        if exe_file.exists():
            if os.path.getmtime(exe_file) > pre_compile_time:
                new_exe = True

        # Compilation complete so remove and close the compile log handler.
        logger.removeHandler(compile_log_handler)
        if new_exe:
            # If there was a compilation, store the log file
            shutil.copy2(compile_log_tempfilename, compile_log_filename)
        compile_log_handler.close()

    # Write compile_kwargs.
    with open(compile_kwargs_filename, 'wb') as f:
        pickle.dump(compile_kwargs, f, protocol=4)

    # ==================== #
    # ----- Sampling ----- #
    # ==================== #

    # Identify reason for resampling.
    resampling_reasons = []
    if force_resample:
        resampling_reasons.append('force resample')
    if new_exe:
        resampling_reasons.append('model recompiled')
    if not samples_filename.exists():
        resampling_reasons.append('samples cache missing')
    if not sample_kwargs_filename.exists():
        resampling_reasons.append('sample_kwargs cache missing')
    # Data cache is allowed to be missing if no data is supplied, since user just wants to return samples rather than
    # check whether resampling is required based on change in data.
    if not data_file.exists() and data is not None:
        resampling_reasons.append('data cache missing')

    # If there is no reason to resample yet then try loading the cached samples.
    if len(resampling_reasons) == 0:

        # Check to see if the given sample_kwargs is the same as the cached sample_kwargs.
        with open(sample_kwargs_filename, 'rb') as f:
            sample_kwargs_same = (sample_kwargs == pickle.load(f))

        # Check to see if the given data is the same as the cached data.
        # If data is not supplied assume data is the same as the cached samples.
        data_same = data is None
        if data_file.exists():
            with open(data_file, 'rb') as f:
                data_cache = pickle.load(f)
                try:
                    np.testing.assert_equal(data, data_cache)
                    data_same = True
                except AssertionError:
                    data_same = False

        # If all checks passed then we can load the cached samples.
        if sample_kwargs_same and data_same:
            with open(samples_filename, 'rb') as f:
                print('Returning cached samples')
                return pickle.load(f), False

        # otherwise, append the reason for failed checks.
        else:
            if not sample_kwargs_same:
                resampling_reasons.append('sample_kwargs changed')

    # Report reasons for resampling.
    print(f'Resampling due to: {", ".join(resampling_reasons)}')

    # If we have reached this point it means some kwargs changed or there are no samples, either
    # way we need data to resample.
    if data is None:
        raise RuntimeError('No data supplied for sampling')

    # We need to recompile, so prepare the sample log file.
    sample_log_handler = logging.FileHandler(filename=sample_log_filename, mode='w')
    sample_log_handler.setLevel(logging.DEBUG)
    logger.addHandler(sample_log_handler)

    # Perform sampling.
    sampling_start = datetime.datetime.now()
    result = m.sample(data=data, **{**sample_kwargs, **other_sample_kwargs})
    sampling_end = datetime.datetime.now()
    print(f'Sampling complete in: {sampling_end - sampling_start}')

    # Sampling complete so remove and close the sample log handler.
    logger.removeHandler(sample_log_handler)
    sample_log_handler.close()

    # Write samples.
    print(f'Pickling posterior...', end='')
    start = datetime.datetime.now()
    samples = result.stan_variables()
    with open(samples_filename, 'wb') as f:
        pickle.dump(samples, f, protocol=4)
    print(f'done ({datetime.datetime.now() - start})')

    # Writing sample_kwargs and data cache is done only after writing the samples cache file so that we know that the
    # caches and supplied values match.

    # Write sample_kwargs.
    with open(sample_kwargs_filename, 'wb') as f:
        pickle.dump(sample_kwargs, f, protocol=4)

    # Write data cache.
    if data is not None:
        with open(data_file, 'wb') as f:
            pickle.dump(data, f, protocol=4)

    # ================================= #
    # ----- Diagnosis and summary ----- #
    # ================================= #

    # TODO report to stdout the number of divergences and any other diagnostic problems.
    #  - Even better would be to structured diagnostic output in something like json format to a file,
    #    but only showing problems such as rhats larger than certain values.
    #  - We would like to ask something like has_problems() and get a bool.

    # Overview and args.
    summary = f'sampling started: {sampling_start}\n' \
              f'sampling ended  : {sampling_end}\n' \
              f'sampling elapsed: {sampling_end - sampling_start}\n\n' \
              f'stan_kwargs={pprint.pformat(sample_kwargs)}\n\n'

    # Diagnosis.
    print(f'Diagnosing...', end='')
    start = datetime.datetime.now()
    summary += f'{result.diagnose()}\n\n'
    print(f'done ({datetime.datetime.now() - start})')

    # Variable summary.
    print(f'Summarising variables...', end='')
    start = datetime.datetime.now()
    df_summary = result.summary()
    df_summary_non_nan = df_summary.dropna()
    df_summary_nan = df_summary[df_summary.isnull().any(axis=1)]
    summary += f'Non-NaN summary:\n{df_summary_non_nan.to_string()}\n\n'
    summary += f'NaN summary:\n{df_summary_nan.to_string()}'
    print(f'done ({datetime.datetime.now() - start})')

    with open(output_dirname / 'summary.txt', 'w') as f:
        f.write(summary)

    # ======================== #
    # ----- Plot summary ----- #
    # ======================== #

    fig = plot_ranks(result)
    fig.savefig(ranks_plot_filename)
    plt.close(fig)

    print(f'output_dir = {output_dirname}')

    return samples, True


def plot_ranks(result, num_worst=10, num_best=5, nbins=None):
    df_summary = result.summary().dropna()

    # Cannot plot more variables that what we have.
    num_worst = min(num_worst, len(df_summary))
    num_best = min(num_best, len(df_summary))

    df_worst = df_summary.sort_values(by='N_Eff').iloc[:num_worst]
    df_best = df_summary.sort_values(by='N_Eff', ascending=False).iloc[:num_best]
    df_ranks = pd.concat((df_worst, df_best))
    nchains = result.chains
    neffs = df_summary['N_Eff'].values
    rhats = df_summary['R_hat'].values

    draws_pd = result.draws_pd()
    nsamples = len(draws_pd)

    if nbins is None:
        nbins = max(3, min(30, int(nsamples / nchains / 10)))

    bin_edges = np.linspace(0, nsamples, nbins + 1)

    usetex = plt.rcParams['text.usetex']
    plt.rcParams['text.usetex'] = False

    fig = plt.figure(figsize=(2 * nchains + 1, 2 * (len(df_ranks) + 2) + 1))
    gs = gridspec.GridSpec(len(df_ranks) + 2, nchains)

    for i, (column_name, neff) in enumerate(zip(df_ranks.index, df_ranks['N_Eff'])):
        print(f'plotting {column_name} {i + 1}/{len(df_ranks)}', flush=True)
        x = draws_pd[column_name].values
        ranks = scipy.stats.rankdata(x)
        for j, r in enumerate(np.split(ranks, nchains)):
            ax = fig.add_subplot(gs[i, j], facecolor='0.95')
            if j == 0:
                ax.set_ylabel(f'{column_name}\nneff={int(neff)}')
            ax.hist(r, bins=bin_edges, zorder=10)
            ax.axhline(len(r) / nbins, c='k', ls=':', zorder=20)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=(j == 0))

    # N_Eff
    print(f'plotting n_eff', flush=True)
    ax = fig.add_subplot(gs[-2, :])
    ax.hist(neffs[np.isfinite(neffs)], bins=100)
    ax.set_xlabel('Number of effective samples')
    ax.set_ylabel('Param Count')

    # Rhat
    print(f'plotting Rhat', flush=True)
    ax = fig.add_subplot(gs[-1, :])
    ax.hist(rhats[np.isfinite(rhats)], bins=100)
    ax.set_xlabel('Rhat')
    ax.set_ylabel('Param Count')

    fig.tight_layout()

    plt.rcParams['text.usetex'] = usetex

    return fig
