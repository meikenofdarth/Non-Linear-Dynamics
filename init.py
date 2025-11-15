# =============================================================================
#
#  (Your full Python script)
#
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm

class PredatorPreyChemostat:
    # (No changes to __init__ or _model_equations)
    def __init__(self, bc=3.3, Kc=4.3, bB=2.25, KB=15, m=0.055, lam=0.4, eps=0.25, delta=0.6, Ni=80.0):
        self.params = {
            'bc': bc, 'Kc': Kc, 'bB': bB, 'KB': KB,
            'm': m, 'lam': lam, 'eps': eps, 'delta': delta, 'Ni': Ni
        }

    def _model_equations(self, t, y):
        N, C, R, B = y
        # Prevent negative populations from breaking the solver
        C = max(0, C)
        R = max(0, R)
        B = max(0, B)
        p = self.params
        Fc = p['bc'] * N / (p['Kc'] + N)
        # Add a small epsilon to the denominator to prevent division by zero if C is exactly -KB
        Fb = p['bB'] * C / (p['KB'] + C + 1e-9)
        dN_dt = p['delta'] * (p['Ni'] - N) - Fc * C
        dC_dt = Fc * C - (Fb * B / p['eps']) - p['delta'] * C
        dR_dt = Fb * R - (p['delta'] + p['m'] + p['lam']) * R
        dB_dt = Fb * R - (p['delta'] + p['m']) * B
        return [dN_dt, dC_dt, dR_dt, dB_dt]

    def run_simulation(self, y0, t_span, t_eval, rtol=1e-6, atol=1e-9): # <<< MODIFICATION
        # sol = solve_ivp(
        #     self._model_equations, t_span, y0,
        #     t_eval=t_eval, method='RK45',
        #     rtol=rtol, atol=atol # <<< MODIFICATION
        # )
        sol = solve_ivp(
            self._model_equations, t_span, y0,
            t_eval=t_eval, method='LSODA', # <<< CHANGE SOLVER HERE TOO
            rtol=rtol, atol=atol
        )
        
        # Check if the integration was successful. status=0 is success.
        if sol.status != 0:
            # Return an empty dictionary to signal failure
            return {} 
            
        return {
            'time': sol.t, 'N': sol.y[0], 'Chlorella': sol.y[1],
            'Reproducing_Brachionus': sol.y[2], 'Total_Brachionus': sol.y[3]
        }

    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
            else:
                print(f"Warning: Parameter '{key}' not found.")

# (No changes to plot_dynamics_for_delta)
def plot_dynamics_for_delta(delta_values, Ni_val, y0, t_sim):
    print("Generating Time Series and Phase Portrait plots...")
    fig, axes = plt.subplots(len(delta_values), 2, figsize=(12, 4 * len(delta_values)), squeeze=False)
    t_eval = np.linspace(t_sim[0], t_sim[1], 2000)
    for i, delta in enumerate(delta_values):
        system = PredatorPreyChemostat(delta=delta, Ni=Ni_val)
        results = system.run_simulation(y0, t_sim, t_eval)
        # ... (rest of the function is the same)
        ax1 = axes[i, 0]
        ax1.plot(results['time'], results['Chlorella'], label='Chlorella (Prey)', color='g')
        ax1.set_title(f'Time Series with δ = {delta}')
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Prey Conc. (C)', color='g')
        ax1.tick_params(axis='y', labelcolor='g')
        ax1.grid(True)
        ax1b = ax1.twinx()
        ax1b.plot(results['time'], results['Total_Brachionus'], label='Brachionus (Predator)', color='k')
        ax1b.set_ylabel('Predator Conc. (B)', color='k')
        ax1b.tick_params(axis='y', labelcolor='k')
        ax2 = axes[i, 1]
        ax2.plot(results['Chlorella'], results['Total_Brachionus'], color='darkblue', lw=0.7)
        ax2.set_title(f'Phase Portrait with δ = {delta}')
        ax2.set_xlabel('Prey Concentration (Chlorella)')
        ax2.set_ylabel('Predator Concentration (Brachionus)')
        ax2.grid(True)
    plt.tight_layout()
    plt.show()

def plot_bifurcation_diagram(delta_range, Ni_val, y0, t_transient, t_measure):
    print("\nGenerating Bifurcation Diagram (this may take a moment)...")
    delta_points = []
    predator_extrema = []
    
    for delta in tqdm(np.linspace(delta_range[0], delta_range[1], 400)):
        system = PredatorPreyChemostat(delta=delta, Ni=Ni_val)
        
        t_span_transient = (0, t_transient)
        # We don't need evaluated points for the transient run, solver can run faster
        # sol_transient = solve_ivp(system._model_equations, t_span_transient, y0, method='RK45')
        sol_transient = solve_ivp(system._model_equations, t_span_transient, y0, method='LSODA') # <<< CHANGE SOLVER


        # <<< MODIFICATION 2: Check for failure after the transient run >>>
        if sol_transient.status != 0:
            # print(f"Solver failed during transient for delta={delta:.3f}. Skipping.")
            continue # Skip to the next delta value

        y0_attractor = sol_transient.y[:, -1]
        
        # <<< MODIFICATION 3: Add a check to prevent negative initial conditions for the next run >>>
        if np.any(y0_attractor < 0):
            # print(f"Unphysical state after transient for delta={delta:.3f}. Skipping.")
            continue # Skip to the next delta value

        t_span_measure = (0, t_measure)
        t_eval_measure = np.linspace(t_span_measure[0], t_span_measure[1], 2000)
        
        # The run_simulation method now handles failures internally
        results = system.run_simulation(y0_attractor, t_span_measure, t_eval_measure)
        
        # <<< MODIFICATION 4: Check if the simulation result is valid >>>
        if not results: # An empty dictionary evaluates to False
            # print(f"Solver failed during measurement for delta={delta:.3f}. Skipping.")
            continue # Skip to the next delta value

        predator_min = np.min(results['Total_Brachionus'])
        predator_max = np.max(results['Total_Brachionus'])
        
        if np.isclose(predator_min, predator_max, atol=1e-2):
            delta_points.append(delta)
            predator_extrema.append(predator_max)
        else:
            delta_points.extend([delta, delta])
            predator_extrema.extend([predator_min, predator_max])

    plt.figure(figsize=(10, 6))
    plt.plot(delta_points, predator_extrema, 'k', alpha=0.5)
    plt.title(f'Bifurcation Diagram for Predator Population (Ni = {Ni_val})')
    plt.xlabel('Dilution Rate (δ)')
    plt.ylabel('Equilibrium / Min-Max Predator Conc. (B)')
    plt.grid(True)
    plt.ylim(0, 60)
    plt.show()

# (No changes to the main execution block)
if __name__ == '__main__':
    initial_conditions = [60.0, 10.0, 5.0, 5.0] 
    Ni_constant = 80.0
    
    delta_values_to_test = [0.25, 0.95, 1.24]
    simulation_time = (0, 200)
    plot_dynamics_for_delta(delta_values_to_test, Ni_constant, initial_conditions, simulation_time)

    # bifurcation_delta_range = (0.1, 1.5)
    # plot_bifurcation_diagram(bifurcation_delta_range, Ni_constant, initial_conditions, t_transient=400, t_measure=200)
    bifurcation_delta_range = (0.1, 1.5)
    # <<< MODIFICATION: Increase transient time significantly
    plot_bifurcation_diagram(
        bifurcation_delta_range, 
        Ni_constant, 
        initial_conditions, 
        t_transient=1000, # Increased from 400 to 1000
        t_measure=400      # Increased measurement time as well
    )