functions {
  vector exp_rolloff(vector fs, real f0, real df) {
    return 1.0 ./ (1.0 + exp((fs - f0)/df));
  }

  vector chirp_freqs(vector ts, real Mc, real tc, real max_freq) {
    vector[num_elements(ts)] fs;

    for (i in 1:num_elements(fs)) {
      if (ts[i] >= tc) {
	fs[i] = max_freq;
      } else {
	real f;

	f = (1100.0*(4.9e-6*Mc)^(5.0/3.0)*(tc - ts[i]))^(-3.0/8.0);
	if (f >= max_freq) {
	  fs[i] = max_freq;
	} else {
	  fs[i] = f;
	}
      }
    }

    return fs;
  }

  vector chirp_amps(vector fs, real A100, real dlnAdlnf) {
    vector[num_elements(fs)] amps;

    for (i in 1:num_elements(fs)) {
      amps[i] = A100*(fs[i]/100.0)^dlnAdlnf;
    }

    return amps;
  }
}

data {
  int nt;
  vector[nt] ts;
  vector[nt] h_obs;
  real tphi;
}

parameters {
  real<lower=0> A100;
  real<lower=0> dlnAdlnf;

  real<lower=0, upper=2.0*pi()> phi;
  real tc;
  
  real<lower=0> Mc;
  real<lower=100.0> max_freq;
  real<lower=0> tau;
  
  real<lower=0> sigma;
}

transformed parameters {
  vector[nt] h_model;

  {
    vector[nt] amp;
    vector[nt] fs;

    fs = chirp_freqs(ts, Mc, tc, max_freq);

    amp = chirp_amps(fs, A100, dlnAdlnf);

    h_model = (1.0 - exp_rolloff(fs, 30.0, 2.0)) .* amp .* cos(phi + 2.0*pi()*fs .* (ts-tphi)) .* exp_rolloff(ts, tc, tau);
  }
}

model {
  h_obs ~ normal(h_model, sigma);

  /* Priors */
  A100 ~ lognormal(0.0, 1.0);
  dlnAdlnf ~ lognormal(0.0, 1.0);

  /* Flat in phi100 */
  tc ~ normal(0.0, 0.01);

  Mc ~ lognormal(log(30.0), 0.2);
  max_freq ~ lognormal(log(250.0), 0.5);
  tau ~ lognormal(log(5e-3), log(2.0));

  sigma ~ lognormal(0.0, 1.0);
}
