functions {
  vector exp_rolloff(vector fs, real f0, real df) {
    return 1.0 ./ (1.0 + exp((fs - f0)/df));
  }

  vector chirp_freqs(vector ts, real Mc, real tc, real max_freq) {
    vector[num_elements(ts)] fs;
    real C;
    real tM;
    real tM53;

    C = (8.0*pi())^(8.0/3.0) / 5.0;
    tM = 4.9e-6*Mc; /* s */
    tM53 = tM^(5.0/3.0); /* s^(5/3) */

    for (i in 1:num_elements(fs)) {
      if (ts[i] >= tc) {
	fs[i] = max_freq;
      } else {
	real f;

	f = (C*tM53*(tc - ts[i]))^(-3.0/8.0);
	if (f >= max_freq) {
	  fs[i] = max_freq;
	} else {
	  fs[i] = f;
	}
      }
    }

    return fs;
  }

  real toff(real f, real Mc, real tc) {
    real C;
    real tM;
    real tM53;

    C = (8.0*pi())^(8.0/3.0) / 5.0;
    tM = 4.9e-6*Mc;
    tM53 = tM^(5.0/3.0);
    
    return tc - f^(-8.0/3.0)/(C*tM53);
  }

  vector chirp_phases(vector ts, real Mc, real tc, real max_freq) {
    vector[num_elements(ts)] phis;
    real C;
    real tM;
    real tM53;
    real tm;
    real phim;
    real t100;
    real phi100;
    
    C = (8.0*pi())^(8.0/3.0) / 5.0;
    tM = 4.9e-6*Mc;
    tM53 = tM^(5.0/3.0);

    tm = toff(max_freq, Mc, tc);
    t100 = toff(100.0, Mc, tc);
    phim = 2.0*pi()*8.0/5.0*(C*tM53)^(-3.0/8.0)*(tc - tm)^(5.0/8.0);
    phi100 = 2.0*pi()*8.0/5.0*(C*tM53)^(-3.0/8.0)*(tc-t100)^(5.0/8.0);

    for (i in 1:num_elements(ts)) {
      if (ts[i] < tm) {
	phis[i] = 2.0*pi()*8.0/5.0*(C*tM53)^(-3.0/8.0)*(tc - ts[i])^(5.0/8.0);
      } else {
	phis[i] = phim + 2.0*pi()*max_freq*(tm - ts[i]);
      }
    }

    /* Phase set to be zero at f = 100 Hz. */
    return phis - phi100;
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
}

parameters {
  real<lower=0> A100;
  real<lower=0> dlnAdlnf;

  real<lower=0, upper=2.0*pi()> phi; /* Phase at 100 Hz, since
					chirp_phase sets zero at f =
					100. */
  real tc;
  
  real<lower=0> Mc;
  real<lower=100.0> max_freq;
  real<lower=0> tau;
  
  real<lower=0> sigma;
}

transformed parameters {
  vector[nt] h_model;

  {
    real tm;
    vector[nt] amp;
    vector[nt] phis;
    vector[nt] fs;

    tm = toff(max_freq, Mc, tc);
    fs = chirp_freqs(ts, Mc, tc, max_freq);
    phis = chirp_phases(ts, Mc, tc, max_freq);
    amp = chirp_amps(fs, A100, dlnAdlnf);

    /* Model rolls "on" at f = (30 +/- 2) Hz and "off" at time of f = fmax with width tau. */
    h_model = (1.0 - exp_rolloff(fs, 30.0, 2.0)) .* amp .* cos(phi - phis) .* exp_rolloff(ts, tm, tau);
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
