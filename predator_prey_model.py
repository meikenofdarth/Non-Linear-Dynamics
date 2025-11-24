# predator_prey_model.py

import numpy as np
from scipy.integrate import solve_ivp

# --- CLASS 1: The Original Model ---
class PredatorPreyChemostat:
    """
    A class to encapsulate the predator-prey model in a chemostat,
    based on the paper "Crossing the Hopf Bifurcation in a Live Predator-Prey System."
    """
    def __init__(self, bc=3.3, Kc=4.3, bB=2.25, KB=15, m=0.055, lam=0.4, eps=0.25, delta=0.6, Ni=80.0):
        self.params = {
            'bc': bc, 'Kc': Kc, 'bB': bB, 'KB': KB,
            'm': m, 'lam': lam, 'eps': eps, 'delta': delta, 'Ni': Ni
        }

    def _model_equations(self, t, y):
        N, C, R, B = y
        C = max(0, C); R = max(0, R); B = max(0, B)
        p = self.params
        Fc = p['bc'] * N / (p['Kc'] + N + 1e-9)
        Fb = p['bB'] * C / (p['KB'] + C + 1e-9)
        dN_dt = p['delta'] * (p['Ni'] - N) - Fc * C
        dC_dt = Fc * C - (Fb * B / p['eps']) - p['delta'] * C
        dR_dt = Fb * R - (p['delta'] + p['m'] + p['lam']) * R
        dB_dt = Fb * R - (p['delta'] + p['m']) * B
        return [dN_dt, dC_dt, dR_dt, dB_dt]

    def run_simulation(self, y0, t_span, t_eval, rtol=1e-6, atol=1e-9):
        sol = solve_ivp(
            self._model_equations, t_span, y0,
            t_eval=t_eval, method='LSODA',
            rtol=rtol, atol=atol
        )
        if sol.status != 0: return {}
        return {
            'time': sol.t, 'N': sol.y[0], 'Chlorella': sol.y[1],
            'Reproducing_Brachionus': sol.y[2], 'Total_Brachionus': sol.y[3]
        }

# --- CLASS 2: The New Variable Quality Model (ADD THIS TO THE FILE) ---
class PredatorPreyVariableQuality(PredatorPreyChemostat):
    """
    An extended model where prey (algal) quality decreases at high densities.
    Inherits from the base class and overrides the model equations.
    """
    def __init__(self, bB_max=2.25, quality_decay=0.05, **kwargs):
        """
        Initializes the model with new parameters for variable quality.
        'bB_max' is the birth rate with high-quality prey.
        'quality_decay' controls how fast the quality drops with density.
        """
        super().__init__(**kwargs)
        # We replace the original bB with bB_max and add the new param
        self.params['bB'] = bB_max 
        self.params['quality_decay'] = quality_decay

    def _model_equations(self, t, y):
        N, C, R, B = y
        C = max(0, C); R = max(0, R); B = max(0, B)
        p = self.params

        # --- KEY MODIFICATION ---
        # The predator's effective birth rate now depends on prey density C.
        bB_effective = p['bB'] / (1 + p['quality_decay'] * C)
        
        Fc = p['bc'] * N / (p['Kc'] + N + 1e-9)
        # Use the new bB_effective in the predator's functional response
        Fb = bB_effective * C / (p['KB'] + C + 1e-9)
        
        dN_dt = p['delta'] * (p['Ni'] - N) - Fc * C
        dC_dt = Fc * C - (Fb * B / p['eps']) - p['delta'] * C
        dR_dt = Fb * R - (p['delta'] + p['m'] + p['lam']) * R
        dB_dt = Fb * R - (p['delta'] + p['m']) * B
        
        return [dN_dt, dC_dt, dR_dt, dB_dt]