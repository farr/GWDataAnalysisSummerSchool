data {
  int npts;
  vector[npts] xs;
  vector[npts] ys;
  vector[npts] sigma_ys;
}

parameters {
  real m;
  real b;
  real<lower=0.1, upper=10.0> ilyas_idiot_factor;
}

model {
  ilyas_idiot_factor ~ lognormal(log(1.0), 1.0);
  ys ~ normal(m*xs + b, sigma_ys*ilyas_idiot_factor);
}
