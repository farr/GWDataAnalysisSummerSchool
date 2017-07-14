functions {
  real dNdz(real z, real N0, real alpha, real beta) {
    return N0*(1.0+z)^alpha*exp(-z/beta);
  }

  real pdet(real ztrue) {
    return 0.5*erfc(5.0*sqrt(2.0)*(log(ztrue) - log(5))/ztrue);
  }

  real N_integrand(real z, real[] state, real[] params, real[] x_r, int[] x_i) {
    real N0;
    real alpha;
    real beta;
    
    N0 = params[1];
    alpha = params[2];
    beta = params[3];

    return pdet(z)*dNdz(z, N0, alpha, beta);
  }
}

data {
  int nobs;
  real zobs[nobs];
  real sigma[nobs];
}

transformed data {
  real x_r[0];
  int x_i[0];
}

parameters {
  real<lower=0> N0;
  real<lower=0> alpha;
  real<lower=0> beta;

  real ztrues[nobs];
}

model {
  real params[3];
  real Nex;
  real integration_result[1,1];
  real zout[1];
  real state0[1];

  params[1] = N0;
  params[2] = alpha;
  params[3] = beta;

  zout[1] = 10.0;
  state0[1] = 0.0;

  integration_result = integrate_ode_rk45(N_integrand, state0, 0.0, zout, params, x_r, x_i);
  Nex = integration_result[1,1];

  for (i in 1:nobs) {
    zobs[i] ~ lognormal(log(ztrues[i]), sigma[i]);
    target += log(dNdz(ztrues[i], N0, alpha, beta));
  }
  target += -Nex;

  /* Priors */
  N0 ~ lognormal(log(10.0), 1.0);
  alpha ~ lognormal(log(2.0), 1.0);
  beta ~ lognormal(log(2.0), 1.0);
}
