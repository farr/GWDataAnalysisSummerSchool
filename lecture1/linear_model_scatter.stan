data {
  int npts;
  vector[npts] xs;
  vector[npts] ys;
  vector[npts] sigma_ys;
}

parameters {
  real m;
  real b;
  real<lower=0> sigma;
  vector[npts] y_true;
}

model {
  y_true ~ normal(m*xs+b, sigma);
  ys ~ normal(y_true, sigma_ys);
}
