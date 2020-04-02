from abc import ABC, abstractmethod
from collections import OrderedDict
import datetime
import difflib
import functools
from hashlib import md5
import inspect
import os
from pathlib import Path
import pickle
import re
from typing import List
import sys

from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import scipy.linalg
import patsy
import pystan


class GPModel(ABC):

    def __init__(self, df, fe_mu_formula=None, re_mu_formulas=None, fe_noise_formula=None, re_noise_formulas=None,
                 priors=None):

        self.df = df
        self.fe_mu_formula     = fe_mu_formula     or '1'
        self.re_mu_formulas    = re_mu_formulas    or []
        self.fe_noise_formula  = fe_noise_formula  or '1'
        self.re_noise_formulas = re_noise_formulas or []

        self.priors = (priors or dict()).copy()  # priors will be modified, so copy

        # init variables that will hold frequencies once set_data is called
        self.freqs = None

        # set default priors
        varnames = ['lambda_noise', 'lambda_gamma', 'lambda_beta',
                    'tau_sigma', 'tau_gamma', 'tau_beta', 'sigma_noise']
        for varname in varnames:
            self.priors[varname] = self.priors.get(varname, 'gamma(2, 1)')

        # create design matrices
        self.x, self.zs, self.dmat_mu   , self.dmats_mu_re    = make_design_matrices(df, fe_mu_formula   , re_mu_formulas   )
        self.w, self.us, self.dmat_noise, self.dmats_noise_re = make_design_matrices(df, fe_noise_formula, re_noise_formulas)

        # setup simplified coefficient and level names to facilitate plotting (this might change when fit is added)
        self.fe_mu_coeffs = self.dmat_mu.design_info.column_names
        self.fe_noise_coeffs = self.dmat_noise.design_info.column_names
        self.re_mu_coeffs = dict()
        self.re_mu_levels = dict()
        self.re_noise_coeffs = dict()
        self.re_noise_levels = dict()
        for re_formula in self.re_mu_formulas:
            re_dmat, factor_dmat = self.dmats_mu_re[re_formula]
            self.re_mu_coeffs[re_formula] = re_dmat.design_info.column_names
            self.re_mu_levels[re_formula] = factor_dmat.design_info.column_names
        for re_formula in self.re_noise_formulas:
            re_dmat, factor_dmat = self.dmats_noise_re[re_formula]
            self.re_noise_coeffs[re_formula] = re_dmat.design_info.column_names
            self.re_noise_levels[re_formula] = factor_dmat.design_info.column_names

        # create stan input data
        self.stan_input_data = OrderedDict()
        self.stan_input_data['N'] = self.x.shape[0]
        self.stan_input_data['P'] = self.x.shape[1]
        self.stan_input_data['Q'] = self.w.shape[1]
        self.stan_input_data['X'] = self.x
        self.stan_input_data['W'] = self.w

        # get subclass-specific variables
        template = self.get_template()
        self.params = self.get_params_fe().copy()
        params_re = self.get_params_re()

        # extract template blueprints
        blueprints = dict(onecol=dict(), multicol=dict())
        locs = dict(data            ='/*** {edge} data {colspec} ***/',
                    params          ='/*** {edge} parameters {colspec} ***/',
                    xfrm_param_decs ='/*** {edge} transformed parameter declarations {colspec} ***/',
                    xfrm_param_defs ='/*** {edge} transformed parameter definitions {colspec} ***/',
                    model           ='/*** {edge} model {colspec} ***/',
                    genqaunt_decs   ='/*** {edge} generated quantities declarations {colspec} ***/',
                    genqaunt_defs   ='/*** {edge} generated quantities definitions {colspec} ***/')
        blueprint_line_idxs = {k: [None, None] for k in locs.keys()}
        for colspec in blueprints.keys():
            for loc, loc_str in locs.items():

                # extract blueprint
                str_start = loc_str.format(edge='start', colspec=colspec)
                str_end   = loc_str.format(edge='end',   colspec=colspec)
                line_start = next(line_idx for line_idx, line in enumerate(template) if str_start in line)
                line_end   = next(line_idx for line_idx, line in enumerate(template) if str_end   in line)
                blueprint = '\n'.join(line for line in template[line_start+1:line_end-1] if len(line.strip()) > 0)
                blueprints[colspec][loc] = blueprint

                # keep line indexes to know which parts of the template to replace with generated code
                if colspec == 'onecol':
                    blueprint_line_idxs[loc][0] = line_start
                else:
                    blueprint_line_idxs[loc][1] = line_end

        # make random-effect code from blueprints to inject back into template
        syringe = {k: [] for k, _ in locs.items()}
        prior_varnames = ['lambda', 'sigma']
        for zs, dpar, pre_vname in [(self.zs, 'eta', 'mu'), (self.us, 'log_omega', 'noise')]:
            for ti, (z, term) in enumerate(zs, 1):

                v = f'{pre_vname}_b{ti}'  # variable name
                nlev, _, ncol = z.shape   # num levels and columns in the random-effect design matrix

                # add default priors for specific terms
                for varname in prior_varnames:
                    prior_name_generic = f'{varname}_{pre_vname}_'
                    prior_name_specific = f'{prior_name_generic}b{ti}'
                    if prior_name_specific not in self.priors:
                        self.priors[prior_name_specific] = self.priors.get(prior_name_generic, 'gamma(2, 1)')

                self.params.append(v)
                self.params += [param.format(v) for param in params_re]
                if ncol == 1:
                    self.stan_input_data[f'Z_{v}'] = z[:, :, 0].T
                else:
                    self.stan_input_data[f'Z_{v}'] = z
                    self.params += [f'chol_corr_{v}']

                blueprint_kwargs = dict(ncol=ncol, nlev=nlev, v=v, term=term, dpar=dpar)
                for loc, blueprint in blueprints['onecol' if ncol == 1 else 'multicol'].items():
                    syringe[loc].append(blueprint.format(**blueprint_kwargs))

        # inject code
        self.code = template.copy()
        for k in syringe.keys():
            contents = '\n'.join(syringe[k])  # combine list into a single string
            line_start, line_end = blueprint_line_idxs[k]
            self.code[line_start] = contents
            self.code[line_start+1:line_end+1] = [None, ] * (line_end - line_start)
        self.code = '\n'.join([line for line in self.code if line is not None])

        # inject priors
        for varname, prior in self.priors.items():

            # check if varname represents a generic prior specification, in which case ignore it
            if varname.endswith('_'):
                continue

            locator = f'prior_{varname}'
            if locator not in self.code:
                raise RuntimeError(f'unknown prior key: {varname}')
            self.code = self.code.replace(locator + ';', prior + ';')

    def __str__(self):
        s = ''

        # plot general sizes
        if 'y' in self.stan_input_data:
            y = self.stan_input_data['y']
            s += f'num units = {y.shape[0]}\n'
            s += f'response shape = {np.array(y.shape[1:]).squeeze()}\n'

        # plot design matrix schemas
        s += f'fe_mu_formula:\n  {self.fe_mu_formula}\n    = '
        s += ' + '.join(simplify_patsy_column_names(self.fe_mu_coeffs)) + '\n'
        s += f'fe_noise_formula:\n  {self.fe_noise_formula}\n    = '
        s += ' + '.join(simplify_patsy_column_names(self.fe_noise_coeffs)) + '\n'
        if len(self.re_mu_formulas) > 0:
            s += 're_mu_formulas:\n  '
            for formula in self.re_mu_formulas:
                coeffs = simplify_patsy_column_names(self.re_mu_coeffs[formula])
                levels = simplify_patsy_column_names(self.re_mu_levels[formula])
                s += f'{formula}\n    = ' + ' + '.join(coeffs)
                s += '\n    | ' + ' + '.join(levels)
        if len(self.re_noise_formulas) > 0:
            s += 're_noise_formulas:\n  '
            for formula in self.re_noise_formulas:
                coeffs = simplify_patsy_column_names(self.re_noise_coeffs[formula])
                levels = simplify_patsy_column_names(self.re_noise_levels[formula])
                s += f'{formula}\n    = ' + ' + '.join(coeffs)
                s += '\n    | ' + ' + '.join(levels)

        return s

    @classmethod
    def get_template(cls) -> List[str]:
        pass

    @classmethod
    def get_params_fe(cls) -> List[str]:
        return ['beta', 'gamma', 'offset_eta', 'sigma_noise', 'lambda_beta', 'lambda_gamma',
                'tau_beta', 'tau_gamma', 'tau_sigma', 'noise']

    @classmethod
    def get_params_re(cls) -> List[str]:
        pass

    @abstractmethod
    def set_data(self, y, **kwargs):
        pass

    @abstractmethod
    def _plot_axes(self, ax, samples, freq_cpm, icpt=False, vmin=None, vmax=None, alpha=0.05, value_label=None):
        pass

    @abstractmethod
    def interp_samples_subdivide(self, samples, subdivision):
        pass

    def sample(self, outpath, ignore_data_change=False, **kwargs):

        if not os.path.exists(outpath):
            os.makedirs(outpath)

        model_cache_filename   = os.path.join(outpath, 'stan_model.pkl')
        model_code_filename    = os.path.join(outpath, 'model.stan')
        samples_cache_filename = os.path.join(outpath, 'samples.pkl')
        diagnostics_filename   = os.path.join(outpath, 'diagnostics.txt')
        traces_filename        = os.path.join(outpath, 'traces.png')

        # write out stan source code
        with open(model_code_filename, 'w') as f:
            f.write(self.code)

        # compile stan code
        stan_model = None
        code_digest = md5(self.code.encode('utf8')).hexdigest()
        if model_cache_filename is not None and os.path.exists(model_cache_filename):
            with open(model_cache_filename, 'rb') as f:
                cached_digest, cached_stan_code, stan_model = pickle.load(f)
            if cached_digest != code_digest:
                stan_model = None
                print('Stan model code is different to the cached version (- cached, + curr):')
                # print the diff between cached and current code
                result = difflib.unified_diff(cached_stan_code.splitlines(), self.code.splitlines(), n=0, lineterm='')
                print('\n'.join(list(result)[2:]))  # [2:] is to remove the control lines '---' and '+++'
                print('recompiling...')
        if stan_model is None:
            compile_start = datetime.datetime.now()
            stan_model = pystan.StanModel(model_code=self.code)
            print(f'Elapsed compilation time: {datetime.datetime.now() - compile_start}')
            if model_cache_filename is not None:
                with open(model_cache_filename, 'wb') as f:
                    pickle.dump((code_digest, self.code, stan_model), f, protocol=2)

        # check if we need to resample, or can load from cache
        needs_sample = True
        samples = None
        dict_digest = lambda d: md5(pickle.dumps(OrderedDict(sorted(d.items())))).hexdigest()
        data_digest = _hash_dict_with_numpy_arrays(self.stan_input_data)  # dict_digest doesn't seem to always produce the same hash for the same numpy arrays
        kwargs_digest = dict_digest(kwargs)
        if samples_cache_filename is not None and os.path.exists(samples_cache_filename):
            with open(samples_cache_filename, 'rb') as f:
                cached_data_digest, cached_kwargs_digest, cached_code_digest, samples = pickle.load(f)
            data_changed = (not ignore_data_change) & (cached_data_digest != data_digest)
            kwargs_changed = cached_kwargs_digest != kwargs_digest
            code_changed = cached_code_digest != code_digest
            needs_sample = data_changed or kwargs_changed or code_changed
            if needs_sample:
                changes = ['model changed'] if code_changed else []
                changes += ['data changed'] if data_changed else []
                changes += ['stan kwargs changed'] if kwargs_changed else []
                print(f'{", ".join(changes)}: resampling...')

        # sample posterior
        if needs_sample:

            # perform sampling
            timer_start = datetime.datetime.now()
            print(f'Started MCMC at {timer_start}')
            stan_fit = stan_model.sampling(data=self.stan_input_data, pars=self.params, **kwargs)
            print(f'Elapsed MCMC {datetime.datetime.now() - timer_start}')

            samples = stan_fit.extract()

            # store diagnostics in samples dict
            if stan_fit is not None:
                samples.seed = stan_fit.get_seed()
                samples.elapsed_time = datetime.datetime.now() - timer_start
                samples.check_fit = str(stan_fit)
                samples.check_treedepth = check_treedepth(stan_fit)
                samples.check_energy = check_energy(stan_fit)
                samples.check_div = check_div(stan_fit)
                samples.sampler_params = stan_fit.get_sampler_params()
                summary = stan_fit.summary()
                samples.n_eff = summary['summary'][:, summary['summary_colnames'].index('n_eff')]
                samples.rhat = summary['summary'][:, summary['summary_colnames'].index('Rhat')]
                print('\n'.join([samples.check_treedepth, samples.check_energy, samples.check_div]))

            # cache samples
            with open(samples_cache_filename, 'wb') as f:
                pickle.dump((data_digest, kwargs_digest, code_digest, samples), f, protocol=2)

            # write various meta info
            with open(diagnostics_filename, 'w') as f:
                f.write(f'seed: {samples.seed}\n')
                f.write(f'elapsed_time: {samples.elapsed_time}\n')
                f.write(f'check_treedepth: {samples.check_treedepth}\n')
                f.write(f'check_divergences: {samples.check_div}\n\n')
                f.write(f'check_fit:\n{samples.check_fit}\n')

            # plot parameter traces
            pars = [p for p in self.params if len(samples[p].shape) < 3]
            fig = plt.figure(figsize=(12, 2 * len(pars) + 3))
            gs = gridspec.GridSpec(len(pars) + 2, 2, width_ratios=[5, 1])
            for i, k in enumerate(pars):
                ax = fig.add_subplot(gs[i, 0])
                is_multi = len(samples[k].shape) > 1
                ax.plot(samples[k], lw=1, alpha=0.6 if is_multi else 0.9)
                ax.set_ylabel(k)
                ax = fig.add_subplot(gs[i, 1])
                if is_multi:
                    for j in range(samples[k].shape[1]):
                        ax.hist(samples[k][:, j], bins=30, alpha=0.7)
                else:
                    ax.hist(samples[k], bins=50)

            # plot n_eff
            ax = fig.add_subplot(gs[-2, :])
            ax.hist(samples.n_eff[np.isfinite(samples.n_eff)], bins=100)
            ax.set_xlabel('Number of effective samples')
            ax.set_ylabel('Param Count')

            # plot Rhat
            ax = fig.add_subplot(gs[-1, :])
            ax.hist(samples.rhat[np.isfinite(samples.rhat)], bins=100)
            ax.set_xlabel('Rhat')
            ax.set_ylabel('Param Count')

            fig.tight_layout()
            fig.savefig(traces_filename, dpi=150)
            plt.close(fig)

        print(f'min, max Rhat = {np.nanmin(samples.rhat)}, {np.nanmax(samples.rhat)}')

        return samples, needs_sample

    def fit(self, samples, component, preds=None, **kwargs_preds):
        if component == 'eta':
            return self._fit(samples['beta'], self.dmat_mu, preds, **kwargs_preds) + samples['offset_eta'][:, np.newaxis]
        elif component == 'omega':
            return self._fit(samples['gamma'], self.dmat_noise, preds, **kwargs_preds)
        else:
            raise RuntimeError(f'unknown component "{component}", components should be "eta" or "omega"')

    # posterior predictive
    def pred(self, samples, preds=None, **kwargs_preds):

        # eta fixed effects
        eta = self._fit(samples['beta'], self.dmat_mu, preds, **kwargs_preds) + samples['offset_eta'][:, np.newaxis]

        # eta random effects
        for ti, (re_term, (dmat_re, _)) in enumerate(self.dmats_mu_re.items(), 1):
            eta += self._fit(samples[f'new_mu_b{ti}'], dmat_re, preds, **kwargs_preds)

        # omega fixed effects
        log_omega = self._fit(samples['gamma'], self.dmat_noise, preds, **kwargs_preds)

        # omega random effects
        for ti, (re_term, (dmat_re, _)) in enumerate(self.dmats_noise_re.items(), 1):
            log_omega += self._fit(samples[f'new_noise_b{ti}'], dmat_re, preds, **kwargs_preds)

        # noise
        eta += samples['noise'] * np.exp(log_omega)

        return eta

    @staticmethod
    def _fit(samples, dmat, preds=None, **kwargs_preds):

        # allow preds kwarg to override kwargs in case some preds have the same name as kwargs of _fit
        if preds is not None:
            kwargs_preds = preds

        # add default data: 0 for numerical or the first category for categorical
        factor_codes = []
        for factorinfo in dmat.design_info.factor_infos.values():
            factorcode = factorinfo.factor.code
            factor_codes.append(factorcode)
            factorvalue = None
            if factorinfo.type == 'numerical':
                if factorcode not in kwargs_preds.keys():
                    factorvalue = 0
            else:  # categorical
                if factorcode not in kwargs_preds.keys():
                    factorvalue = factorinfo.categories[0]
            if factorvalue is not None:
                kwargs_preds[factorcode] = factorvalue
                print(f'fit adding \'{factorcode}\' = {factorvalue}')

        if len(factor_codes) == 0:
            # trivial design matrix is just intercept
            return samples[:, 0, ...]
        else:

            # keep only relevent factors
            preds_actual = dict()
            for k, v in kwargs_preds.items():
                if k in factor_codes:
                    preds_actual[k] = v

            # create prediction dmat
            dmat_predict = np.array(patsy.build_design_matrices([dmat.design_info], preds_actual))[0][0]

            # return np.einsum('sk..., k -> s...', samples, dmat_predict)
            return np.einsum('ska, k -> sa', samples, dmat_predict)

    def plot(self, freq_cpm, decs, eqns, titles=None, icpt_tx=None, diff_tx=None, alpha=0.05, simplify_coeffs=True,
             icpt_value_label=None, diff_value_label='Ratio', offset_eta=None, force_diff=False):
        """
        Plot results.
        @param freq_cpm: frequency in log2 cpm.
        @param decs: either coefficients such as samples['beta'] or samples['gamma'], or a dict str->samples containing
                     named variables that are accessed by the equations in eqns.
        @param eqns: list of lists specifying a row-major matrix of equations as str, where variables' values are
                     looked up in the decs parameter, where each equation is plotted on a single axes.
        @param titles: list of lists of the same size as eqns, specifying the titles of the axes specified in eqns.
        @param icpt_tx: intercept transform, for example, if log responses are used can be np.exp to transform back.
        @param diff_tx: comparison transform. Note: by default the identity is used where a non-intercept equation
                        is considered a difference on the log-scale, and thus a ratio is plotted.
        @param alpha: value for plotting "significant" contours or band lines.
        @param simplify_coeffs: whether change the name of coefficients to be more human readable.
        @param icpt_value_label: axis label referring to intercept-based values.
        @param diff_value_label: axis label referring to comparison-based values, defaults to "Ratio".
        @param offset_eta: only supply if decs doesn't contain samples from fit, which already include offset_eta.
        @param force_diff: whether to force considering all values as comparisons.
        @return: matplotlib figure
        """

        # default transforms are identities
        icpt_tx = icpt_tx or (lambda x: x)
        diff_tx = diff_tx or (lambda x: x)

        # no decs dict specified, so decs must be ndarray, where decs.shape[1] == len(eqns)
        is_marginal = type(decs) is not dict
        if is_marginal:
            assert decs.shape[1] == len(eqns)
            if simplify_coeffs:
                eqns = simplify_patsy_column_names(eqns)
            keys = eqns
            values = [decs[:, i, ...] for i in range(len(keys))]
            if titles is None:
                titles = list(keys)

            # generate a new set of keys for dec_list such that they can be used in eval, keeping 'Intercept'
            i = 0
            new_keys = []
            for key in keys:
                if key == 'Intercept':
                    new_keys.append(key)
                else:
                    new_keys.append(f'a{i}')
                    i += 1

            # recreate decs with new keys
            decs = dict(zip(new_keys, values))
            eqns = new_keys

            titles = to_grid_list(titles)
            eqns = to_grid_list(eqns)

        # test for 1d or 2d data
        # Note: 2d freq-phase is flattenend into shape (nsamples, nfreq * nphase),
        #       but freq_freq is (nsamples, nfreq, nfreq)
        samples0 = next(iter(decs.values()))
        if samples0.shape[-1] == len(freq_cpm) and len(samples0.shape) == 2:
            extrema_func = lambda x: x
        else:
            extrema_func = lambda x: np.median(x, axis=0)

        nrows = len(eqns)
        ncols = max(map(len, eqns))

        fig = plt.figure(figsize=(ncols * 4.2, nrows * 3.35))
        gs = gridspec.GridSpec(nrows, ncols)

        # TODO too memory intensive, don't cache samples, recompute each time?

        # first pass: parse equations, calculate extrema and generate axes
        cells = []
        icpt_min, icpt_max = np.inf, -np.inf
        diff_min, diff_max = np.inf, -np.inf
        for ri, row_eqn in enumerate(eqns):
            if row_eqn is None or len(row_eqn) == 0:
                continue
            for ci, cell_eqn in enumerate(row_eqn):
                if cell_eqn is None or cell_eqn.strip() == '':
                    continue

                if force_diff:
                    is_icpt = False
                else:
                    # identify intercept vs diff
                    if is_marginal:
                        is_icpt = cell_eqn == 'Intercept'
                    else:
                        is_icpt = ('-' not in cell_eqn)

                samples_cond = eval(cell_eqn, {}, decs)

                # calculate extrema
                if is_icpt:
                    if offset_eta is not None:
                        samples_cond = samples_cond + offset_eta[:, np.newaxis]
                    samples_cond = icpt_tx(samples_cond)
                    samples_cond_reduced = extrema_func(samples_cond)
                    icpt_min = min(icpt_min, np.min(samples_cond_reduced))
                    icpt_max = max(icpt_max, np.max(samples_cond_reduced))
                else:
                    samples_cond = diff_tx(samples_cond)
                    samples_cond_reduced = extrema_func(samples_cond)
                    diff_min = min(diff_min, np.min(samples_cond_reduced))
                    diff_max = max(diff_max, np.max(samples_cond_reduced))

                # create axes, and add title
                ax = fig.add_subplot(gs[ri, ci])
                ax.set_title(cell_eqn if titles is None else titles[ri][ci])

                cells.append((ax, (cell_eqn, decs), is_icpt))

        diff_extrema = max(abs(diff_min), abs(diff_max))

        # second pass: plot
        for ax, (cell_eqn, decs), is_icpt in cells:

            if is_icpt:
                vmin, vmax = icpt_min, icpt_max
                value_label = icpt_value_label
            else:
                vmin, vmax = -diff_extrema, diff_extrema
                value_label = diff_value_label

            # recompute sum so that keeping previously calculated sum doesn't use up memory
            samples_cond = eval(cell_eqn, {}, decs)
            if is_icpt:
                if offset_eta is not None:
                    samples_cond = samples_cond + offset_eta[:, np.newaxis]
                samples_cond = icpt_tx(samples_cond)
            else:
                samples_cond = diff_tx(samples_cond)

            # plot
            self._plot_axes(ax, samples_cond, freq_cpm, icpt=is_icpt, vmin=vmin, vmax=vmax, alpha=alpha,
                            value_label=value_label)

        fig.tight_layout()
        return fig

    def plot_coeffs(self, samples, outpath):

        # model.freqs is on the log2 scale
        if self.freqs is None:
            raise RuntimeError('set_data must be called on model before calling plot_coeffs')
        freqs_cpm = 2 ** self.freqs

        # plot betas
        fig = self.plot(freqs_cpm, samples['beta'], self.fe_mu_coeffs, offset_eta=samples['offset_eta'])
        fig.savefig(outpath / 'beta.png', dpi=150)
        plt.close(fig)

        # plot gamma
        fig = self.plot(freqs_cpm, samples['gamma'], self.fe_noise_coeffs)
        fig.savefig(outpath / 'gamma.png', dpi=150)
        plt.close(fig)

        # plot mu random effects
        if len(self.re_mu_formulas) > 0:
            print(f'Plotting random-effects mu')
        for ri, re_formula in enumerate(self.re_mu_formulas):
            b = samples[f'mu_b{ri + 1}']
            label = re_formula.split('|')[-1].strip()

            coeffs = simplify_patsy_column_names(self.re_mu_coeffs[re_formula])
            levels = simplify_patsy_column_names(self.re_mu_levels[re_formula])

            print(f'  {re_formula}: {coeffs}')

            for ci, coeff in enumerate(coeffs):
                if len(coeffs) > 1:
                    b_coeff = b[:, :, ci, :]
                else:
                    b_coeff = b
                fig = self.plot(freqs_cpm, b_coeff, levels)
                fig.savefig(outpath / f'mu_{label}_{coeff}.png', dpi=150)
                plt.close(fig)

        # plot noise random effects
        if len(self.re_noise_formulas) > 0:
            print(f'Plot random-effects noise:')
        for ri, re_formula in enumerate(self.re_noise_formulas):
            b = samples[f'noise_b{ri + 1}']
            label = re_formula.split('|')[-1].strip()

            coeffs = simplify_patsy_column_names(self.re_noise_coeffs[re_formula])
            levels = simplify_patsy_column_names(self.re_noise_levels[re_formula])

            print(f'  {re_formula}: {coeffs}')

            for ci, coeff in enumerate(coeffs):
                if len(coeffs) > 1:
                    b_coeff = b[:, :, ci, :]
                else:
                    b_coeff = b
                fig = self.plot(freqs_cpm, b_coeff, levels)
                fig.savefig(outpath / f'noise_{label}_{coeff}.png', dpi=150)
                plt.close(fig)


class GPFreqModel(GPModel):

    def __init__(self, df, fe_mu_formula=None, re_mu_formulas=None, fe_noise_formula=None, re_noise_formulas=None,
                 priors=None):
        super().__init__(df, fe_mu_formula, re_mu_formulas, fe_noise_formula, re_noise_formulas, priors)

    @classmethod
    def get_template(cls):
        this_module = sys.modules[cls.__module__]
        codepath = Path(inspect.getfile(this_module)).parent / 'gp1d_template.stan'
        with codepath.open('r') as f:
            return f.read().splitlines()

    @classmethod
    def get_params_fe(cls):
        return super().get_params_fe() + ['lambda_noise']

    @classmethod
    def get_params_re(cls):
        return ['sigma_{}', 'lambda_{}', 'new_{}']

    def set_data(self, y, **kwargs):
        freqs = kwargs['freqs']
        self.stan_input_data['y'] = y
        self.stan_input_data['F'] = len(freqs)
        self.stan_input_data['f'] = freqs

        self.freqs = freqs

    @classmethod
    def _interp(cls, x, x_star, y, per_sample_kernel_func_x, nugget_size=1e-6):

        nsamples, n = y.shape

        nugget_x = nugget_size * np.identity(len(x))

        y_star = np.empty((nsamples, len(x_star)))
        for i in range(nsamples):

            kern_x      = per_sample_kernel_func_x(i, x     [:, None], x[None, :])
            kern_x_star = per_sample_kernel_func_x(i, x_star[:, None], x[None, :])

            # note: no need to scale by variance here since it cancels out through the solve followed by multiply
            # i.e: chol_solve(K alpha1 == y/a)  and then   y_star = a (L* alpha1)   is the same as
            #      chol_solve(K alpha2 == y  )  and then   y_star =    L* alpha2    with alpha2 = a * alpha1

            kern_chol = np.linalg.cholesky(kern_x + nugget_x)
            alpha = scipy.linalg.cho_solve((kern_chol, True), y[i])

            y_star[i, ...] = (kern_x_star @ alpha).flatten()

        return y_star

    @classmethod
    def interp_samples(cls, samples, x, x_star):

        def sqr_exp_kernel(s, l, x1, x2):
            return s**2 * np.exp(-0.5 * ((x1 - x2) / l)**2)

        lengthscales_x = dict(beta=samples['lambda_beta'], gamma=samples['lambda_gamma'])
        nsamples = len(lengthscales_x['beta'])
        nx_star = len(x_star)

        samples_star = dict()

        def kern_func_x(prm, k, i, x1, x2):
            return sqr_exp_kernel(1, lengthscales_x[prm][i, k], x1, x2)

        # interpolate fixed-effects
        for param in ['beta', 'gamma']:

            v = samples[param]
            _, ncols, _ = v.shape

            # interpolate param
            v_star = np.empty((nsamples, ncols, nx_star))
            for j in range(ncols):
                kern_x = functools.partial(kern_func_x, param, j)
                v_star[:, j, :] = cls._interp(x, x_star, v[:, j, ...], kern_x)

            samples_star[param] = v_star

        # copy over non-x-dependent parameters
        for key in samples.keys():
            if key not in samples_star.keys():
                samples_star[key] = samples[key].copy()

        return samples_star

    def interp_samples_subdivide(self, samples, subdivision: int):
        log2_freqs_cpm = self.freqs
        nfreq = len(log2_freqs_cpm)
        log2_freqs_cpm_subdiv = np.linspace(log2_freqs_cpm[0], log2_freqs_cpm[-1], (nfreq - 1) * subdivision + 1)
        samples_subdiv = self.interp_samples(samples, log2_freqs_cpm, log2_freqs_cpm_subdiv)
        return samples_subdiv, log2_freqs_cpm_subdiv

    def _plot_axes(self, ax, samples, freq_cpm, icpt=False, vmin=None, vmax=None, alpha=0.05, value_label=None):

        log2_freqs_cpm = np.log2(freq_cpm)

        # create ticks, labels, and grid for freq and phase
        freq_order1 = int(round(log2_freqs_cpm[0]))
        freq_order2 = int(round(log2_freqs_cpm[-1]))
        freq_ticks = np.arange(freq_order1, freq_order2 + 1, dtype=np.int32)
        freq_labels = ['{}'.format(2**fe) if fe >= 0 else '1/{}'.format(2**(-fe)) for fe in freq_ticks]

        ax.get_figure().sca(ax)

        lower = 100 * (0.5 * alpha)
        upper = 100 * (1 - 0.5 * alpha)

        # if we have 3 dimensions then it is a freq-freq comparison
        if len(samples.shape) == 3:

            cdict = dict(
                blue =[(0, 0, 0.5), (0.25, 1  , 1  ), (0.5, 1, 1), (0.75, 0  , 0  ), (1, 0  , 0)],
                green=[(0, 0, 0  ), (0.25, 0.4, 0.4), (0.5, 1, 1), (0.75, 0.4, 0.4), (1, 0  , 0)],
                red  =[(0, 0, 0  ), (0.25, 0  , 0  ), (0.5, 1, 1), (0.75, 1  , 1  ), (1, 0.5, 0)],
            )
            cmap_diff = mcolors.LinearSegmentedColormap('RedBlue', cdict, N=501)

            xegrid, yegrid = edge_meshgrid(log2_freqs_cpm, log2_freqs_cpm)  # for pcolormesh
            xcgrid, ycgrid = np.meshgrid(log2_freqs_cpm, log2_freqs_cpm)    # for contour

            level = 1 - 0.5 * alpha
            samples_pos = np.mean(samples > 0, axis=0)
            samples_neg = np.mean(samples < 0, axis=0)
            samples = np.median(samples, axis=0)
            samples[(samples_pos < level) & (samples_neg < level)] = 0

            mappable = ax.pcolormesh(xegrid, yegrid, samples.T, vmin=vmin, vmax=vmax, cmap=cmap_diff)
            cbar = ax.get_figure().colorbar(mappable, use_gridspec=True, label=value_label)

            kwargs = dict(colors='k', linewidths=1, levels=[level])
            if np.min(samples_pos) < level < np.max(samples_pos):
                ax.contour(xcgrid, ycgrid, samples_pos.T, linestyles='-', **kwargs)
            if np.min(samples_neg) < level < np.max(samples_neg):
                ax.contour(xcgrid, ycgrid, samples_neg.T, linestyles='-', **kwargs)

            # colorbar ticks
            log2_vmax = np.log2(np.exp(vmax))

            # logarithmically-spaced ticks
            if log2_vmax > 1:
                cbar_ticks = np.arange(0, np.ceil(log2_vmax) + 1)
                cbar_ticks = np.r_[-cbar_ticks[1:][::-1], cbar_ticks]
                cbar_ticklabels = [f'{2 ** v:g}' if v >= 0 else f'{2 ** -v:g}⁻¹' for v in cbar_ticks]
                cbar_ticks = np.log(2 ** cbar_ticks)

            # linearly-spaced ticks
            else:
                ticker = mticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10])
                cbar_ticks = np.log2(np.array(ticker.tick_values(1, np.exp(vmax))))
                cbar_ticks = np.r_[-cbar_ticks[1:][::-1], cbar_ticks]
                cbar_ticklabels = [f'{2 ** v:g}' if v >= 0 else f'{2 ** -v:g}⁻¹' for v in cbar_ticks]
                cbar_ticks = np.log(2 ** cbar_ticks)

            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_ticklabels)

            ax.set_xlabel('Frequency (cpm) [X]')
            ax.set_ylabel('Frequency (cpm) [Y]')
            ax.set_xticks(freq_ticks)
            ax.set_xticklabels(freq_labels)
            ax.set_yticks(freq_ticks)
            ax.set_yticklabels(freq_labels)

        # otherwise just a normal amplitude over frequency plot
        else:

            mu_ci = np.array([np.percentile(samples, q, axis=0) for q in (lower, upper)])

            facecolor_value = mcolors.rgb_to_hsv(ax.get_facecolor()[:3])[-1]
            if facecolor_value > 0.5:
                linecolor = 'k'
            else:
                linecolor = 'w'
            ax.plot(log2_freqs_cpm, mu_ci.T, c=linecolor, ls=':', zorder=40)
            ax.plot(log2_freqs_cpm, samples.T, c=linecolor, lw=0.5, alpha=0.01, zorder=20)

            if not icpt:
                ax.axhline(0, c=(1, 0, 0), zorder=50)

            ax.set_xlabel('Frequency (cpm)')
            ax.set_ylabel(value_label)

            # ratio ticks
            if not icpt:
                log2_vmax = np.log2(np.exp(vmax))

                # logarithmically-spaced ticks
                if log2_vmax > 2:
                    ticks = np.arange(0, np.ceil(log2_vmax) + 1)
                    ticks = np.r_[-ticks[1:][::-1], ticks]
                    ticklabels = [f'{2**v:g}' if v >= 0 else f'{2**-v:g}⁻¹' for v in ticks]
                    ticks = np.log(2**ticks)

                # linearly-spaced ticks
                else:
                    ticker = mticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10])
                    ticks = np.log2(np.array(ticker.tick_values(1, np.exp(vmax))))
                    ticks = np.r_[-ticks[1:][::-1], ticks]
                    ticklabels = [f'{2**v:g}' if v >= 0 else f'{2**-v:g}⁻¹' for v in ticks]
                    ticks = np.log(2**ticks)

                ax.set_yticks(ticks)
                ax.set_yticklabels(ticklabels)

            ax.set_ylim(vmin, vmax)

            ax.set_xticks(freq_ticks)
            ax.set_xticklabels(freq_labels)


class GPFreqPhaseModel(GPModel):

    def __init__(self, df, fe_mu_formula=None, re_mu_formulas=None, fe_noise_formula=None, re_noise_formulas=None,
                 priors=None, sep=None):
        super().__init__(df, fe_mu_formula, re_mu_formulas, fe_noise_formula, re_noise_formulas, priors)
        self.sep = sep

        # init variable that will hold phases once set_data is called
        self.phases = None

    @classmethod
    def get_template(cls):
        this_module = sys.modules[cls.__module__]
        codepath = Path(inspect.getfile(this_module)).parent / 'gp2d_template.stan'
        with codepath.open('r') as f:
            return f.read().splitlines()

    @classmethod
    def get_params_fe(cls):
        return super().get_params_fe() + ['lambda_noise', 'lambda_rho_beta', 'lambda_rho_gamma', 'lambda_rho_noise']

    @classmethod
    def get_params_re(cls):
        return ['sigma_{}', 'lambda_{}', 'lambda_rho_{}', 'new_{}']

    def set_data(self, y, **kwargs):
        freqs = kwargs['freqs']
        phases = kwargs['phases']

        self.freqs = freqs
        self.phases = phases

        # convert from N,F,H major->minor order to N,F*H reshape used in the Stan code (with F*H being F-minor)
        self.stan_input_data['y'] = y.transpose((2, 1, 0)).reshape(-1, y.shape[0]).T

        self.stan_input_data['F'] = len(freqs)
        self.stan_input_data['H'] = len(phases)
        self.stan_input_data['f'] = freqs
        self.stan_input_data['h'] = phases

    def _plot_axes(self, ax, samples, freq_cpm, icpt=False, vmin=None, vmax=None, alpha=0.05, value_label=None):
        log2_freqs_cpm = np.log2(freq_cpm)

        samples = np.reshape(samples, (-1, samples.shape[-1] // len(freq_cpm), len(freq_cpm)))

        nphase = samples.shape[1]
        phase_edges = np.linspace(-np.pi, np.pi, nphase + 1)
        phases = edges_to_centers(phase_edges)

        # create 1x3 periodic grid so that contours are drawn correctly
        phases_1x3 = periodic_coord_wings(phases)

        # pad freqs so that contours at top and bottom are treated properly
        delta_f = log2_freqs_cpm[1] - log2_freqs_cpm[0]
        log2_freqs_cpm = np.r_[log2_freqs_cpm[0] - delta_f, log2_freqs_cpm, log2_freqs_cpm[-1] + delta_f]
        samples = np.pad(samples, [(0, 0), (0, 0), (1, 1)], mode='edge')

        def tile_1x3(x):
            return np.r_[x, x, x]

        hegrid, fegrid = edge_meshgrid(phases_1x3, log2_freqs_cpm)
        hcgrid, fcgrid = np.meshgrid(phases_1x3, log2_freqs_cpm)

        # create ticks, labels, and grid for freq and phase
        freq_order1 = int(log2_freqs_cpm[0])
        freq_order2 = int(log2_freqs_cpm[-1])
        freq_ticks = np.arange(freq_order1, freq_order2 + 1, dtype=np.int32)
        freq_labels = ['{}'.format(2**fe) if fe >= 0 else '1/{}'.format(2**(-fe)) for fe in freq_ticks]
        phase_ticks = np.pi * np.array([-1, -0.5, 0, 0.5, 1])
        phase_labels = ['-π', '-π/2', '0', 'π/2', 'π']

        # cmap for difference-based samples
        # cdict = dict(
        #     blue =[(0, 0, 0.5), (0.25, 1  , 1  ), (0.5, 1, 1), (0.75, 0  , 0  ), (1, 0  , 0)],
        #     green=[(0, 0, 0  ), (0.25, 0.4, 0.4), (0.5, 1, 1), (0.75, 0.4, 0.4), (1, 0  , 0)],
        #     red  =[(0, 0, 0  ), (0.25, 0  , 0  ), (0.5, 1, 1), (0.75, 1  , 1  ), (1, 0.5, 0)],
        # )
        cdict = dict(
            blue =[(0, 0, 1.0), (0.3, 1  , 1  ), (0.5, 0, 0), (0.7, 0.2, 0.2), (1, 0.6, 0)],
            green=[(0, 0, 0.8), (0.3, 0.3, 0.3), (0.5, 0, 0), (0.7, 0.3, 0.3), (1, 0.8, 0)],
            red  =[(0, 0, 0.6), (0.3, 0.2, 0.2), (0.5, 0, 0), (0.7, 1  , 1  ), (1, 1.0, 0)],
        )
        cmap_diff = mcolors.LinearSegmentedColormap('RedBlue', cdict, N=501)

        ax.get_figure().sca(ax)

        mappable = ax.pcolormesh(hegrid, fegrid, tile_1x3(np.median(samples, axis=0)).T,
                                 vmin=vmin, vmax=vmax, cmap='viridis' if icpt else cmap_diff)
        cbar = ax.get_figure().colorbar(mappable, use_gridspec=True, label=value_label or '')

        # colorbar ticks
        if not icpt:
            log2_vmax = np.log2(np.exp(vmax))

            # logarithmically-spaced ticks
            if log2_vmax > 1:
                cbar_ticks = np.arange(0, np.ceil(log2_vmax) + 1)
                cbar_ticks = np.r_[-cbar_ticks[1:][::-1], cbar_ticks]
                cbar_ticklabels = [f'{2**v:g}' if v >= 0 else f'{2**-v:g}⁻¹' for v in cbar_ticks]
                cbar_ticks = np.log(2**cbar_ticks)

            # linearly-spaced ticks
            else:
                ticker = mticker.MaxNLocator(nbins=5, steps=[1, 2, 5, 10])
                cbar_ticks = np.log2(np.array(ticker.tick_values(1, np.exp(vmax))))
                cbar_ticks = np.r_[-cbar_ticks[1:][::-1], cbar_ticks]
                cbar_ticklabels = [f'{2**v:g}' if v >= 0 else f'{2**-v:g}⁻¹' for v in cbar_ticks]
                cbar_ticks = np.log(2**cbar_ticks)

            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels(cbar_ticklabels)

        # plot isovelocity lines
        if self.sep is not None:
            velocities = [1, 3, 10, 30, 100]  # cm/min when sep is in cm
            for vel in velocities:
                freq_cpm_edges = centers_to_edges(freq_cpm, log=True)
                vel_phases = 2 * np.pi * self.sep * freq_cpm_edges / vel
                vel_freqs = np.log2(freq_cpm_edges)
                ax.plot(vel_phases, vel_freqs, c='w', lw=1, ls=':', alpha=0.5)
                ax.plot(-vel_phases, vel_freqs, c='w', lw=1, ls=':', alpha=0.5)

        # plot contours at alpha
        level = 1 - 0.5 * alpha
        if not icpt:
            samples_pos = np.mean(samples > 0, axis=0)
            samples_neg = np.mean(samples < 0, axis=0)
            # make padded boundary below contours
            samples_pos[:,  0] = 0.5
            samples_pos[:, -1] = 0.5
            samples_neg[:,  0] = 0.5
            samples_neg[:, -1] = 0.5
            kwargs = dict(colors='w', linewidths=2, levels=[level])
            if np.min(samples_pos) < level < np.max(samples_pos):
                ax.contour(hcgrid, fcgrid, tile_1x3(samples_pos).T, linestyles='-', **kwargs)
            if np.min(samples_neg) < level < np.max(samples_neg):
                ax.contour(hcgrid, fcgrid, tile_1x3(samples_neg).T, linestyles='-', **kwargs)

        # plot contours at alpha for mirror diff
        if icpt:
            # if intercept, then only care about plotting which direction has higher values
            samples_diff_extreme = np.mean((samples - samples[:, ::-1, :]) > 0, axis=0)
        else:
            # if diff, then we care about plotting which direction has significantly larger absolute difference
            samples_diff_extreme = np.mean(((samples - samples[:, ::-1, :]) > 0) & (samples > 0) |
                                           ((samples - samples[:, ::-1, :]) < 0) & (samples < 0), axis=0)
        if np.min(samples_diff_extreme) < level < np.max(samples_diff_extreme):  # only plot if there exists contour
            ax.contour(hcgrid, fcgrid, tile_1x3(samples_diff_extreme).T,
                       linestyles=[(2, (2, 2))], colors='k', linewidths=2, levels=[level])
            ax.contour(hcgrid, fcgrid, tile_1x3(samples_diff_extreme).T,
                       linestyles=[(0, (2, 2))], colors='w', linewidths=2, levels=[level])

        ax.axvline(0, c='w', lw=1, alpha=0.5)

        ax.set_yticks(freq_ticks)
        ax.set_yticklabels(freq_labels)
        ax.set_xticks(phase_ticks)
        ax.set_xticklabels(phase_labels)

        ax.set_xlim(phase_edges[0], phase_edges[-1])  # must limit here since we're drawing at 1x3
        ax.set_ylim(fegrid[1, 0], fegrid[-2, -1])  # remove freq padding

        ax.set_ylabel('Frequency (cpm)')
        ax.set_xlabel('Phase (radians)')

    @classmethod
    def _interp(cls, x, x_star, w, w_star, y, per_sample_kernel_func_x, per_sample_kernel_func_w, nugget_size=1e-6):

        nsamples, n = y.shape

        nugget_x = nugget_size * np.identity(len(x))
        nugget_w = nugget_size * np.identity(len(w))

        y_star = np.empty((nsamples, len(x_star) * len(w_star)))
        for i in range(nsamples):

            kern_x      = per_sample_kernel_func_x(i, x     [:, None], x[None, :])
            kern_x_star = per_sample_kernel_func_x(i, x_star[:, None], x[None, :])

            kern_w      = per_sample_kernel_func_w(i, w     [:, None], w[None, :])
            kern_w_star = per_sample_kernel_func_w(i, w_star[:, None], w[None, :])

            inv_kern_w = np.linalg.inv(kern_w + nugget_w)
            inv_kern_x = np.linalg.inv(kern_x + nugget_x)
            alpha = kron_mvprod([inv_kern_x, inv_kern_w], y[i]).flatten()

            y_star[i, ...] = kron_mvprod([kern_x_star, kern_w_star], alpha).flatten()

        return y_star

    @classmethod
    def interp_samples(cls, samples, x, w, x_star, w_star):

        def sqr_exp_kernel(s, l, x1, x2):
            return s**2 * np.exp(-0.5 * ((x1 - x2) / l)**2)

        def periodic_sqr_exp_kernel(s, l, w1, w2):
            return s**2 * np.exp(-2 * np.sin(0.5 * np.abs(w1 - w2))**2 / l**2)

        lengthscales_x = dict(beta=samples['lambda_beta'][:, 0], gamma=samples['lambda_gamma'][:, 0])
        lengthscales_w = dict(beta=samples['lambda_beta'][:, 1], gamma=samples['lambda_gamma'][:, 1])
        # sigma_noise = samples['sigma_noise']
        nsamples = len(lengthscales_x['beta'])
        nx_star = len(x_star)
        nw_star = len(w_star)

        samples_star = dict()

        def kern_func_x(prm, k, i, x1, x2):
            return sqr_exp_kernel(1, lengthscales_x[prm][i, k], x1, x2)

        def kern_func_w(prm, k, i, w1, w2):
            return periodic_sqr_exp_kernel(1, lengthscales_w[prm][i, k], w1, w2)

        # interpolate fixed-effects
        for param in ['beta', 'gamma']:

            v = samples[param]
            _, ncols, _ = v.shape

            # interpolate param
            v_star = np.empty((nsamples, ncols, nx_star * nw_star))
            for j in range(ncols):
                kern_x = functools.partial(kern_func_x, param, j)
                kern_w = functools.partial(kern_func_w, param, j)
                v_star[:, j, :] = cls._interp(x, x_star, w, w_star, v[:, j, ...], kern_x, kern_w)

            samples_star[param] = v_star

        # copy over non-x-dependent parameters
        for key in samples.keys():
            if key not in samples_star.keys():
                samples_star[key] = samples[key].copy()

        return samples_star

    def interp_samples_subdivide(self, samples, subdivision):

        # subdivide frequencies
        log2_freqs_cpm = self.freqs
        nfreq = len(log2_freqs_cpm)
        log2_freqs_cpm_subdiv = np.linspace(log2_freqs_cpm[0], log2_freqs_cpm[-1], (nfreq - 1) * subdivision + 1)

        # subdivide phases
        phases = self.phases
        nphase = len(phases)
        phase_edges = centers_to_edges(phases)
        phase_edges_subdiv = np.linspace(phase_edges[0], phase_edges[-1], nphase * subdivision)
        phases_subdiv = edges_to_centers(phase_edges_subdiv)

        samples_subdiv = self.interp_samples(samples, log2_freqs_cpm, phases, log2_freqs_cpm_subdiv, phases_subdiv)
        return samples_subdiv, log2_freqs_cpm_subdiv


def make_design_matrices(df, fe_formula=None, re_formulas=None):

    fe_formula  = fe_formula  or '1'
    re_formulas = re_formulas or []

    dmat_fe = patsy.dmatrix(fe_formula, df, eval_env=3)
    x = np.asarray(dmat_fe)

    dmats_re = OrderedDict()

    # TODO pre-process by breaking up factor expressions
    # "(x | g1 / g2)"  ->  "(x | g1) + (x | g1:g2)"
    # "(x || g)"       ->  "(1 | g)  + (0 + x | g)"

    zs = []
    for re_formula in re_formulas:
        re_expr, factor = (a.strip() for a in re_formula.split('|'))

        # each row's level
        factor_dmat = patsy.dmatrix(f'0 + {factor}', df, eval_env=3)
        factor_x = np.asarray(factor_dmat, dtype=int)
        if not np.all(np.sum(factor_x, axis=1) == 1):
            raise Exception(f'Incorrectly specified factor "{factor}" in formula: "{re_formula}"')

        # design matrix of the random effect expression
        re_dmat = patsy.dmatrix(f'{re_expr}', df, eval_env=3)
        re_x = np.asarray(re_dmat)
        dmats_re[re_formula] = (re_dmat, factor_dmat)

        # given the NxP re_x design matrix and the NxL factor_x design matrix,
        # generate an NxPL z matrix where row n is re_x[n,:] (kron) factor_x[n,:], then reshape for Stan
        z = np.einsum('np,nl->lnp', re_x, factor_x).reshape(factor_x.shape[1], re_x.shape[0], re_x.shape[1])

        zs.append((z, re_formula))

    return x, zs, dmat_fe, dmats_re


def simplify_patsy_column_names(column_name, simplify_terms=True):

    if type(column_name) is list:
        return [simplify_patsy_column_names(n, simplify_terms) for n in column_name]

    if column_name != 'Intercept':
        column_name = column_name.replace('T.', '')
        if simplify_terms and '[' in column_name:
            labels = []
            for term in column_name.split(':'):
                if '[' in term:
                    m = re.search(r'.+\[(.+)\]', term)
                    labels.append(m.group(1))
                else:
                    labels.append(term)
            column_name = '_'.join(labels)
    return column_name


def to_grid_list(x: List) -> List[List]:
    nrows = int(np.floor(np.sqrt(len(x))))
    ncols = int(np.ceil(len(x) / nrows))
    x = x + [None, ] * (nrows * ncols - len(x))  # expand x with Nones to make it factorise to nrows * ncols
    return [x[i:i+ncols] for i in range(0, len(x), ncols)]


def periodic_coord_wings(x):
    dx = x[1] - x[0]
    return np.r_[x - x[-1] + x[0] - dx, x, x - x[0] + x[-1] + dx]


def kron_mvprod(ms, v):
    u = v.copy()
    for m in ms[::-1]:
        u = m @ np.reshape(u.T, (m.shape[1], -1))
    return u.T


def _hash_dict_with_numpy_arrays(d):
    m = md5()
    for k, v in sorted(d.items()):
        m.update(pickle.dumps(k))
        if type(v) == np.ndarray:
            m.update(v.tobytes())
        else:
            m.update(pickle.dumps(v))
    return m.hexdigest()


def edges_to_centers(edges, log=False):
    if log:
        edges = np.log2(edges)
    centers = edges[1:] - 0.5 * (edges[1] - edges[0])
    if log:
        centers = 2 ** centers
    return centers


def centers_to_edges(centers, log=False):
    if log:
        centers = np.log2(centers)
    if len(centers) == 1:
        dx = 1
    else:
        dx = centers[1] - centers[0]
    edges = np.r_[centers, centers[-1] + dx] - 0.5 * dx
    if log:
        edges = 2 ** edges
    return edges


def edge_meshgrid(centers_x, centers_y, logx=False, logy=False):
    return np.meshgrid(centers_to_edges(centers_x, logx), centers_to_edges(centers_y, logy))


# the following are based on Michael Betancourt's stan_utility.py:
# https://github.com/betanalpha/jupyter_case_studies/blob/master/pystan_workflow/stan_utility.py

def check_div(fit):
    """Check transitions that ended with a divergence"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    divergent = [x for y in sampler_params for x in y['divergent__']]
    n = int(sum(divergent))
    N = len(divergent)
    ret = f'{n} of {N} iterations ended with a divergence ({100*n/N}%)'
    if n > 0:
        ret += '\nTry running with larger adapt_delta to remove the divergences'
    return ret


def check_treedepth(fit):
    """Check transitions that ended prematurely due to maximum tree depth limit"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    depths = [x for y in sampler_params for x in y['treedepth__']]
    max_depth = int(np.max(depths))
    # n = sum(1 for x in depths if x == max_depth)
    # N = len(depths)
    # ret = f'top tree depth of {max_depth} accounted for {n} of {N} iterations ({100*n/N}%)'
    ret = f'top tree depth of {max_depth}'
    return ret


def check_energy(fit):
    """Checks the energy Bayesian fraction of missing information (E-BFMI)"""
    sampler_params = fit.get_sampler_params(inc_warmup=False)
    ret = []
    for chain_num, s in enumerate(sampler_params):
        energies = s['energy__']
        numer = sum((energies[i] - energies[i - 1])**2 for i in range(1, len(energies))) / len(energies)
        denom = np.var(energies)
        if numer / denom < 0.2:
            ret.append(f'Chain {chain_num}: E-BFMI = {numer / denom}'
                       '\nE-BFMI below 0.2 indicates you may need to reparameterize your model')

    return '\n'.join(ret)

