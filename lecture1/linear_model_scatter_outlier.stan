data {
  int npts;
  vector[npts] xs;
  vector[npts] ys;
  vector[npts] sigma_ys;
}

transformed data {
  real xcent;

  xcent = mean(xs);
}

parameters {
  real<lower=-pi()/2.0, upper=pi()/2.0> theta;
  real b;
  real<lower=0> sigma;
  vector[npts] y_true;
  real<lower=0,upper=1> A;
  real mu;
  real<lower=0> outlier_sigma;
}

transformed parameters {
  real m;

  m = tan(theta);
}

model {
  A ~ beta(3,1); /* <A> = 0.75, sigma_A ~ 1/4 */
  outlier_sigma ~ lognormal(log(400.0), 1.0);
  sigma ~ lognormal(log(30.0), 1.0); /* With by-hand outlier rejection, sigma ~ 30 */
  /* Uniform prior on theta */
  b ~ normal(xcent, 400.0);
  mu ~ normal(xcent, 400.0);
  
  for (i in 1:npts) {
    target += log_mix(A, normal_log(y_true[i], m*xs[i]+b, sigma), normal_log(y_true[i], mu, outlier_sigma));
  }
  ys ~ normal(y_true, sigma_ys);
}
