// Copyright 2023 Lukasz Wiklendt. All rights reserved.
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
}

data {
  int N;  // num rows
  int P;  // num mu       population-effects
  int Q;  // num residual propulation-effects
  int F;  // num frequencies
  int H;  // num phases
  
  matrix[N,P] X;  // mu       population predictors
  matrix[N,Q] W;  // residual population predictors
  
  // observations are in phase-major frequency-minor order, s.t for observation n,
  // y[n,2] is the 2nd frequency of the 1st phase,
  // y[n,1+H] is the 1st frequency of the 2nd phase,
  // and to_matrix(y[n], F, H) produces the correct matrix[F,H] arrangement
  matrix[N,F*H] y;

  // cholesky decomposed kernel matrix for frequencies and phases
  matrix[F,F] kern_chol_f;  
  matrix[H,H] kern_chol_h;
  
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
  real mean_y = mean(y);
}


parameters {
  matrix[P,F*H] z_beta;
  matrix[Q,F*H] z_gamma;
  
  real offset_eta;
  real<lower=0> tau_beta;
  real<lower=0> tau_gamma;
  real<lower=0> sigma_noise;
  
  /*** start parameters onecol ***/
  
  real<lower=0> sigma_{v};
  matrix[{nlev},F*H] z_{v};
  
  /*** end parameters onecol ***/
  /*** start parameters multicol ***/
  
  vector<lower=0>[{ncol}] sigma_{v};
  cholesky_factor_corr[{ncol}] chol_corr_{v};
  matrix[{ncol},F*H] z_{v}[{nlev}];
  
  /*** end parameters multicol ***/
}


transformed parameters {
  
  matrix[P,F*H] beta;
  matrix[Q,F*H] gamma;
  
  /*** start transformed parameter declarations onecol ***/
  
  matrix[{nlev},F*H] {v};
  
  /*** end transformed parameter declarations onecol ***/
  /*** start transformed parameter declarations multicol ***/
  
  matrix[{ncol},F*H] {v}[{nlev}];
  
  /*** end transformed parameter declarations multicol ***/
  
  {
    for (q in 1:Q) {
      gamma[q] = to_row_vector(tau_gamma * kron_mvprod2(kern_chol_h, kern_chol_f, to_matrix(z_gamma[q], F, H)));
    }
    
    for (p in 1:P) {
      beta[p] = to_row_vector(tau_beta * kron_mvprod2(kern_chol_h, kern_chol_f, to_matrix(z_beta[p], F, H)));
    }
    
    /*** start transformed parameter definitions onecol ***/
    
    for (l in 1:{nlev}) {{
      {v}[l] = to_row_vector(sigma_{v} * kron_mvprod2(kern_chol_h, kern_chol_f, to_matrix(z_{v}[l], F, H)));
    }}

    /*** end transformed parameter definitions onecol ***/
    /*** start transformed parameter definitions multicol ***/
    
    // sample each column
    for (c in 1:{ncol}) {{
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
  
  offset_eta ~ student_t(3, mean_y, 4);
  tau_gamma ~ prior_tau_gamma;
  tau_beta  ~ prior_tau_beta;
  sigma_noise ~ prior_sigma_noise;
  
  eta       = X * beta + offset_eta;
  log_omega = W * gamma;
  
  to_vector(z_beta)  ~ normal(0, 1);
  to_vector(z_gamma) ~ normal(0, 1);
  
  /*** start model onecol ***/
  
  sigma_{v}  ~ prior_sigma_{v};
  to_vector(z_{v}) ~ normal(0, 1);
  for (n in 1:N) {{
    int l = l_{v}[n];
    {dpar}[n,] += Z_{v}[n] * {v}[l];
  }}
  
  /*** end model onecol ***/
  /*** start model multicol ***/
  
  sigma_{v} ~ prior_sigma_{v};
  for (n in 1:N) {{
    int l = l_{v}[n];
    to_vector(z_{v}[l]) ~ normal(0, 1);
    {dpar}[n,] += Z_{v}[n,] * {v}[l];
  }}
  
  /*** end model multicol ***/
  
  // calculate residuals (structured)
  {
    to_vector(y) ~ normal(to_vector(eta), exp(to_vector(log_omega)));
  }
}

generated quantities {

  vector[F*H] noise;
  
  /*** start generated quantities declarations onecol ***/
  
  vector[F*H] new_{v};

  /*** end generated quantities declarations onecol ***/
  /*** start generated quantities declarations multicol ***/
  
  matrix[{ncol},F*H] new_{v};
  
  /*** end generated quantities declarations multicol ***/
  
  {
    vector[F*H] new_z;  // for sampling from standard normal
    
    // sample noise for posterior-predictive
    for (i in 1:F*H) {{
      noise[i] = normal_rng(0, sigma_noise);
    }}
    
    /*** start generated quantities definitions onecol ***/
    
    for (i in 1:F*H) {{
      new_z[i] = normal_rng(0, 1);
    }}
    new_{v} = to_vector(kron_mvprod2(kern_chol_h, kern_chol_f, to_matrix(new_z, F, H)));
    
    /*** end generated quantities definitions onecol ***/
    /*** start generated quantities definitions multicol ***/
    
    // sample each column
    for (c in 1:{ncol}) {{
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
