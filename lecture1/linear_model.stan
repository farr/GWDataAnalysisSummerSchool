data {
  int npts;
  vector[npts] xs;
  vector[npts] ys;
  vector[npts] sigma_ys;
}

parameters {
  real m;
  real b;
}

model {
  ys ~ normal(m*xs + b, sigma_ys);
}
