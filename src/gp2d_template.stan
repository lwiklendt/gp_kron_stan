// Copyright 2019 Lukasz Wiklendt. All rights reserved.
// This work is licensed under the terms of the MIT license.  
// For a copy, see <https://opensource.org/licenses/MIT>.

// Templates for generating Stan code.
// The blocks of code enclosed between /*** start ... ***/ and /*** end ... ***/ are to be
// replicated for each independent group/random-effect term, where
// {v} is the grouping term label, which can just be b1, b2, b3, ..., etc,
// {ncol} is the number of columns in the grouping term's design matrix,
// {nlev} is the number of independent levels in the grouping term.

functions {
  // return to_matrix(y, b, a)
  // where y = (A @ B) to_vector(V)
  // and A is axa, B is bxb, V is bxa, and @ is the Kronecker product
  matrix kron_mvprod2(matrix A, matrix B, matrix V) {
    return (A * (B * V)')';
  }
  
  matrix sqrexp(matrix dist, real lengthscale) {
    return exp(-0.5 * (dist .* dist) * inv_square(lengthscale));
  }
  
  matrix kern_func_f(matrix dist, real lengthscale) {
    return sqrexp(dist, lengthscale);
  }
  
  matrix kern_func_h(matrix dist, real lengthscale) {
    return sqrexp(2 * dist, lengthscale);  // scale 2 since periodic kernel is exp(-2 dist^2/l^2), and sqrexp does * -0.5
  }
}

data {
  int N;  // num rows
  int P;  // num mu       population-effects
  int Q;  // num residual propulation-effects
  int F;  // num frequencies
  int H;  // num phases
  
  matrix[N,P] X;  // mu       population predictors
  matrix[N,Q] W;  // residual population predictors
  
  vector[F] f;  // frequency grid locations (log-spaced)
  vector[H] h;  // phase     grid locations (lin-spaced radians for periodic kernel)
  
  // observations are in phase-major frequency-minor order, s.t for observation n,
  // y[n,2] is the 2nd frequency of the 1st phase,
  // y[n,1+H] is the 1st frequency of the 2nd phase,
  // and to_matrix(y[n], F, H) produces the correct matrix[F,H] arrangement
  matrix[N,F*H] y;
  
  /*** start data onecol ***/
  
  // {v} (1|{nlev}): {term}
  vector[N] Z_{v};
  int<lower=1, upper={nlev}> l_{v}[N];
  
  /*** end data onecol ***/
  /*** start data multicol ***/
  
  // {v} ({ncol}|{nlev}): {term}
  matrix[N,{ncol}] Z_{v};
  int<lower=1, upper={nlev}> l_{v}[N];
  
  /*** end data multicol ***/
}


transformed data {
  real nugget = 1e-6;
  matrix[F,F] I_f = diag_matrix(rep_vector(1, F));
  matrix[H,H] I_h = diag_matrix(rep_vector(1, H));
  matrix[F,F] fd =           fabs(rep_matrix(f, F) - rep_matrix(f', F));
  matrix[H,H] hd = sin(0.5 * fabs(rep_matrix(h, H) - rep_matrix(h', H)));
  real mean_y = mean(y);
}


parameters {
  vector[2] z_log_lambda_noise;
  matrix[2,P] z_log_lambda_beta;
  matrix[2,Q] z_log_lambda_gamma;
  
  cholesky_factor_corr[2] Lambda_chol_noise;
  cholesky_factor_corr[2] Lambda_chol_beta;
  cholesky_factor_corr[2] Lambda_chol_gamma;
  
  matrix[P,F*H] z_beta;
  matrix[Q,F*H] z_gamma;
  
  real offset_eta;
  real<lower=0> tau_beta;
  real<lower=0> tau_gamma;
  real<lower=0> tau_sigma;
  real<lower=sqrt(nugget)> sigma_noise;
  
  /*** start parameters onecol ***/
  
  real<lower=0> sigma_{v};
  matrix[{nlev},F*H] z_{v};
  cholesky_factor_corr[2] Lambda_chol_{v};
  vector[2] z_log_lambda_{v};
  
  /*** end parameters onecol ***/
  /*** start parameters multicol ***/
  
  vector<lower=0>[{ncol}] sigma_{v};
  cholesky_factor_corr[{ncol}] chol_corr_{v};
  matrix[{ncol},F*H] z_{v}[{nlev}];
  cholesky_factor_corr[2] Lambda_chol_{v};
  matrix[2,{ncol}] z_log_lambda_{v};
  
  /*** end parameters multicol ***/
}


transformed parameters {
  
  vector[2]   log_lambda_noise = Lambda_chol_noise * z_log_lambda_noise;
  matrix[2,P] log_lambda_beta  = Lambda_chol_beta  * z_log_lambda_beta;
  matrix[2,Q] log_lambda_gamma = Lambda_chol_gamma * z_log_lambda_gamma;
  vector[2]   lambda_noise = exp(log_lambda_noise);
  matrix[2,P] lambda_beta  = rep_matrix(lambda_noise, P) + exp(log_lambda_beta);
  matrix[2,Q] lambda_gamma = rep_matrix(lambda_noise, Q) + exp(log_lambda_gamma);
  
  matrix[P,F*H] beta;
  matrix[Q,F*H] gamma;
  
  /*** start transformed parameter declarations onecol ***/
  
  matrix[{nlev},F*H] {v};
  vector[2] log_lambda_{v} = Lambda_chol_{v} * z_log_lambda_{v};
  vector[2] lambda_{v} = lambda_noise + exp(log_lambda_{v});
  
  /*** end transformed parameter declarations onecol ***/
  /*** start transformed parameter declarations multicol ***/
  
  matrix[{ncol},F*H] {v}[{nlev}];
  matrix[2,{ncol}] log_lambda_{v} = Lambda_chol_{v} * z_log_lambda_{v};
  matrix[2,{ncol}] lambda_{v} = rep_matrix(lambda_noise, {ncol}) + exp(log_lambda_{v});
  
  /*** end transformed parameter declarations multicol ***/
  
  {
    matrix[F,F] kern_chol_f;
    matrix[H,H] kern_chol_h;
    
    for (q in 1:Q) {
      kern_chol_f = cholesky_decompose(kern_func_f(fd, lambda_gamma[1,q]) + nugget * I_f);
      kern_chol_h = cholesky_decompose(kern_func_h(hd, lambda_gamma[2,q]) + nugget * I_h);
      gamma[q] = to_row_vector(tau_gamma * kron_mvprod2(kern_chol_h, kern_chol_f, to_matrix(z_gamma[q], F, H)));
    }
    
    for (p in 1:P) {
      kern_chol_f = cholesky_decompose(kern_func_f(fd, lambda_beta[1,p]) + nugget * I_f);
      kern_chol_h = cholesky_decompose(kern_func_h(hd, lambda_beta[2,p]) + nugget * I_h);
      beta[p] = to_row_vector(tau_beta * kron_mvprod2(kern_chol_h, kern_chol_f, to_matrix(z_beta[p], F, H)));
    }
    
    /*** start transformed parameter definitions onecol ***/
    
    kern_chol_f = cholesky_decompose(kern_func_f(fd, lambda_{v}[1]) + nugget * I_f);
    kern_chol_h = cholesky_decompose(kern_func_h(hd, lambda_{v}[2]) + nugget * I_h);
    for (l in 1:{nlev}) {{
      {v}[l] = to_row_vector(sigma_{v} * kron_mvprod2(kern_chol_h, kern_chol_f, to_matrix(z_{v}[l], F, H)));
    }}

    /*** end transformed parameter definitions onecol ***/
    /*** start transformed parameter definitions multicol ***/
    
    // sample each column
    for (c in 1:{ncol}) {{
      kern_chol_f = cholesky_decompose(kern_func_f(fd, lambda_{v}[1,c]) + nugget * I_f);
      kern_chol_h = cholesky_decompose(kern_func_h(hd, lambda_{v}[2,c]) + nugget * I_h);
      for (l in 1:{nlev}) {{
        {v}[l][c,] = sigma_{v}[c] * to_row_vector(kron_mvprod2(kern_chol_h, kern_chol_f, to_matrix(z_{v}[l][c,], F, H)));
      }}
    }}
    
    // correlate columns
    for (l in 1:{nlev}) {{
        {v}[l] = chol_corr_{v} * {v}[l];
    }}
    
    /*** end transformed parameter definitions multicol ***/
  }
}


model {
  matrix[N,F*H] eta;
  matrix[N,F*H] log_omega;
  
  // user-level priors on lengthscale
  lambda_noise            ~ prior_lambda_noise;
  to_vector(lambda_beta)  ~ prior_lambda_beta;
  to_vector(lambda_gamma) ~ prior_lambda_gamma;
  Lambda_chol_noise ~ lkj_corr_cholesky(2);
  Lambda_chol_beta  ~ lkj_corr_cholesky(2);
  Lambda_chol_gamma ~ lkj_corr_cholesky(2);
  
  // change-of-variables adjustment
  target += sum(log_lambda_noise);
  target += sum(log_lambda_beta);
  target += sum(log_lambda_gamma);
  
  to_vector(z_log_lambda_noise) ~ normal(0, 1);
  to_vector(z_log_lambda_beta)  ~ normal(0, 1);
  to_vector(z_log_lambda_gamma) ~ normal(0, 1);
  
  offset_eta ~ student_t(3, mean_y, 4);
  
  tau_gamma ~ prior_tau_gamma;
  tau_beta  ~ prior_tau_beta;
  tau_sigma ~ prior_tau_sigma;
  sigma_noise ~ prior_sigma_noise;
  
  eta       = X * beta + offset_eta;
  log_omega = W * gamma;
  
  to_vector(z_beta)  ~ normal(0, 1);
  to_vector(z_gamma) ~ normal(0, 1);
  
  /*** start model onecol ***/
  
  sigma_{v}  ~ prior_sigma_{v};
  lambda_{v} ~ prior_lambda_{v};
  z_log_lambda_{v} ~ normal(0, 1);
  target += sum(log_lambda_{v});
  Lambda_chol_{v} ~ lkj_corr_cholesky(2);
  to_vector(z_{v}) ~ normal(0, 1);
  for (n in 1:N) {
    int l = l_{v};
    {dpar}[n,] += Z_{v}[n] * {v}[l];
  }
  
  /*** end model onecol ***/
  /*** start model multicol ***/
  
  sigma_{v} ~ prior_sigma_{v};
  to_vector(lambda_{v}) ~ prior_lambda_{v};
  to_vector(z_log_lambda_{v}) ~ normal(0, 1);
  target += sum(log_lambda_{v});
  Lambda_chol_{v} ~ lkj_corr_cholesky(2);
  for (n in 1:N) {{
    int l = l_{v}[n];
    to_vector(z_{v}[l]) ~ normal(0, 1);
    {dpar}[n,] += Z_{v}[n,] * {v}[l];
  }}
  
  /*** end model multicol ***/
  
  // calculate residuals (structured)
  {
    matrix[F,F] kern_f = tau_sigma^2 * kern_func_f(fd, lambda_noise[1]) + nugget * I_f;
    matrix[H,H] kern_h =               kern_func_h(hd, lambda_noise[2]) + nugget * I_h;
    matrix[F,F] Qf = eigenvectors_sym(kern_f);
    matrix[H,H] Qh = eigenvectors_sym(kern_h);
    vector[F] Lf = eigenvalues_sym(kern_f);
    vector[H] Lh = eigenvalues_sym(kern_h);

    matrix[F,H] eigenvalues = Lf * Lh' + sigma_noise^2;
    for (n in 1:N) {
      matrix[F,H] r = to_matrix((y[n] - eta[n]) .* exp(-log_omega[n]), F, H);
      matrix[F,H] alpha = kron_mvprod2(Qh', Qf', r);
      alpha = alpha ./ eigenvalues;
      alpha = kron_mvprod2(Qh, Qf, alpha);
      target += -0.5 * sum(r .* alpha);
    }

    target += -sum(log_omega);
    target += -0.5 * N * sum(log(eigenvalues));
  }
}

generated quantities {
  real lambda_rho_beta  = multiply_lower_tri_self_transpose(Lambda_chol_beta )[1, 2];
  real lambda_rho_gamma = multiply_lower_tri_self_transpose(Lambda_chol_gamma)[1, 2];
  real lambda_rho_noise = multiply_lower_tri_self_transpose(Lambda_chol_noise)[1, 2];
  vector[F*H] noise;
  
  /*** start generated quantities declarations onecol ***/
  
  real lambda_rho_{v} = multiply_lower_tri_self_transpose(Lambda_chol_{v})[1, 2];
  vector[F*H] new_{v};

  /*** end generated quantities declarations onecol ***/
  /*** start generated quantities declarations multicol ***/
  
  real lambda_rho_{v} = multiply_lower_tri_self_transpose(Lambda_chol_{v})[1, 2];
  matrix[{ncol},F*H] new_{v};
  
  /*** end generated quantities declarations multicol ***/
  
  {
    matrix[F,F] kern_chol_f;
    matrix[H,H] kern_chol_h;
    vector[F*H] new_z;  // for sampling from standard normal
    
    // sample noise for posterior-predictive
    kern_chol_f = cholesky_decompose(kern_func_f(fd, lambda_noise[1]) + sigma_noise^2 * I_f);
    kern_chol_h = cholesky_decompose(kern_func_h(hd, lambda_noise[2]) + sigma_noise^2 * I_h);
    for (i in 1:F*H) {{
      new_z[i] = normal_rng(0, 1);
    }}
    noise = to_vector(tau_sigma * kron_mvprod2(kern_chol_h, kern_chol_f, to_matrix(new_z, F, H)));
    
    /*** start generated quantities definitions onecol ***/
    
    kern_chol_f = cholesky_decompose(kern_func_f(fd, lambda_{v}[1]) + nugget * I_f);
    kern_chol_h = cholesky_decompose(kern_func_h(hd, lambda_{v}[2]) + nugget * I_h);
    for (i in 1:F*H) {{
      new_z[i] = normal_rng(0, 1);
    }}
    new_{v} = to_vector(kron_mvprod2(kern_chol_h, kern_chol_f, to_matrix(new_z, F, H)));
    
    /*** end generated quantities definitions onecol ***/
    /*** start generated quantities definitions multicol ***/
    
    // sample each column
    for (c in 1:{ncol}) {{
      kern_chol_f = cholesky_decompose(kern_func_f(fd, lambda_{v}[1,c]) + nugget * I_f);
      kern_chol_h = cholesky_decompose(kern_func_h(hd, lambda_{v}[2,c]) + nugget * I_h);
      for (i in 1:F*H) {{
        new_z[i] = normal_rng(0, 1);
      }}
      new_{v}[c,] = sigma_{v}[c] * to_row_vector(kron_mvprod2(kern_chol_h, kern_chol_f, to_matrix(new_z, F, H)));
    }}
    
    // correlate columns
    new_{v} = chol_corr_{v} * new_{v};
    
    /*** end generated quantities definitions multicol ***/
  }
}
