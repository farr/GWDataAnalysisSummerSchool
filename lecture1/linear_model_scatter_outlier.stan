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
  real<lower=0,upper=1> A;
  real mu;
  real<lower=0> outlier_sigma;
}

model {
  outlier_sigma ~ exponential(0.005);
  sigma ~ exponential(0.05);
  for (i in 1:npts) {
    target += log_mix(A, normal_log(y_true[i], m*xs[i]+b, sigma), normal_log(y_true[i], mu, outlier_sigma));
  }
  ys ~ normal(y_true, sigma_ys);
}
