functions {
  /* This is our rate function: */
  real dNdz(real z, real N0, real alpha, real beta) {
    return N0*(1.0+z)^alpha*exp(-z/beta);
  }

  /* This is the integrand that will end up giving us N:

     N = I(10)

     where 

     I(x) = \int_0^x dz dN/dz

     so that 

     dI/dx = dN/dz(z = x)

     Here the "state" of the ODE is one-dimensional (it is I(z)), the
     parameters are {N0, alpha, beta}, and there is no real "data",
     x_r, or integer "data", x_i.  (Stan distinguishes between data
     and parameters here because it will compute the gradient of the
     integral with respect to the parameters but not the data; so
     putting things that aren't sampling parameters or derived from
     sampling parameters into the x_r and x_i arrays will save a lot
     of computation of gradients.
*/
  real[] N_integrand(real z, real[] state, real[] params, real[] x_r, int[] x_i) {
    real N0;
    real alpha;
    real beta;
    real dstatedz[1];
    
    N0 = params[1];
    alpha = params[2];
    beta = params[3];

    dstatedz[1] = dNdz(z, N0, alpha, beta);

    return dstatedz;
  }
}

data {
  int nobs; /* The number of observations */
  real zobs[nobs]; /* The redshifts observed. */

  int nmodel; /* The number of points at which the model should be
		 computed and stored (for convenient plotting
		 later). */
  real zs_model[nmodel]; /* The redshift of those points. */
}

transformed data {
  /* We need to declare any x_r and x_i data (real and integer) data
     in the `data` or `transformed data` blocks so Stan knows it
     doesn't depend on any parameters.  Here we just declare some
     zero-element arrays.

  Stan also requires that the limits of integration (or points at
  which you want the ODE output---there can be more than one in more
  complicated cases) be declared as data, because it cannot take
  gradients with respect to them. So, we declare `zout = 10.0` here.*/
  real x_r[0];
  int x_i[0];
  real zout[1];

  zout[1] = 10.0;
}

parameters {
  /* Note that all parameters are constrained to be positive. */
  real<lower=0> N0;
  real<lower=0> alpha;
  real<lower=0> beta;
}


model {
  /* First we set up the necessary bits for solving the ODE and
     extracting the answer into the `Nex` parameter (number of
     expected events).  */
  real Nex;
  real params[3];
  real integration_result[1,1];
  real state0[1];

  params[1] = N0;
  params[2] = alpha;
  params[3] = beta;

  state0[1] = 0.0;

  integration_result = integrate_ode_rk45(N_integrand, state0, 0.0, zout, params, x_r, x_i);
  Nex = integration_result[1,1];

  /* For each observation, we accumulate a factor of dNdz(zobs) */
  for (i in 1:nobs) {
    target += log(dNdz(zobs[i], N0, alpha, beta));
  }
  /* And we normalise by exp(-<N>). */
  target += -Nex;

  /* Broad priors, consistent with true values.  */
  N0 ~ lognormal(log(10.0), 1.0);
  alpha ~ lognormal(log(2.0), 1.0);
  beta ~ lognormal(log(2.0), 1.0);
}

generated quantities {
  /* The generated quantities block is for things that aren't needed
     to compute the model posterior, but that you would like to have
     samples from; the code here is only run when Stan has chosen a
     set of parameters to output, instead of running every time Stan
     takes an internal step "flying" through parameter space. 

     We would like to be able to plot dN/dz for the output parameters,
     so we compute that here. */
  real dNdz_model[nmodel];

  for (i in 1:nmodel) {
    dNdz_model[i] = dNdz(zs_model[i], N0, alpha, beta);
  }
}
