functions {
  vector flare_rate(vector ts, real A, real t0, real dt) {
    vector[num_elements(ts)] rates;

    rates = A ./ (1.0 + exp(-(ts-t0)/(dt/10.0))) .* (1.0 - 1.0 ./ (1.0 + exp(-(ts-t0)/dt)));

    return rates;
  }
  
  vector counts_from_rates(vector ts, vector rates) {
    vector[num_elements(ts)-1] cts;

    for (i in 1:num_elements(cts)) {
      cts[i] = 0.5*(ts[i+1]-ts[i])*(rates[i]+rates[i+1]);
    }

    return cts;
  }
}

data {
  int nc;
  int no;

  vector[nc+1] ts_bin;
  vector[no] bg_rate;
  int counts[no,nc];
}

transformed data {
  vector[no] bg_counts;

  for (i in 1:no) {
    bg_counts[i] = bg_rate[i]*(ts_bin[2]-ts_bin[1]);
  }
}

parameters {
  real mu_A;
  real<lower=0> sigma_A;
  real mu_t;
  real<lower=0> sigma_t;
  real mu_tau;
  real<lower=0> sigma_tau;

  /* It is convenient to actually sample in a parameter space with
     constant prior scale; we can produce the actual parameters via
     re-scaling in the `transformed parameters` block. */
  vector[no] norm_As;
  vector[no] norm_ts;
  vector[no] norm_taus;
}

transformed parameters {
  vector[no] As;
  vector[no] ts;
  vector[no] taus;
  vector[nc+1] flare[no];

  As = exp(mu_A + sigma_A*norm_As);
  ts = mu_t + sigma_t*norm_ts;
  taus = exp(mu_tau + sigma_tau*norm_taus);

  for (i in 1:no) {
    flare[i] = flare_rate(ts_bin, As[i], ts[i], taus[i]);
  }
}

model {
  /* Priors on population */
  /* 95% limits on mu_A are log(0.1) and log(10.0) */
  mu_A ~ normal(0.5*(log(0.1)+log(10.0)), (log(10.0) - log(0.1))/2.0); 
  sigma_A ~ normal(0.0, 1.0);

  mu_t ~ normal(0.0, 5.0);
  sigma_t ~ normal(0.0, 2.0);

  /* 95% limits on mu_tau are log(0.1) and log(1.0) */
  mu_tau ~ normal(0.5*(log(0.1)+log(1.0)), (log(1.0)-log(0.1))/2.0);
  sigma_tau ~ normal(0.0, 1.0);

  /* Population on System */
  norm_As ~ normal(0.0, 1.0);
  norm_ts ~ normal(0.0, 1.0);
  norm_taus ~ normal(0.0, 1.0);

  /* Observations */
  for (i in 1:no) {
    counts[i] ~ poisson(bg_counts[i] + counts_from_rates(ts_bin, flare[i]));
  }
}
