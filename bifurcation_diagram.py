# --- bifurcation_diagram.py ---

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm  # A nice progress bar for long simulations

# Import your existing, well-structured model
from predator_prey_model import PredatorPreyChemostat

def generate_bifurcation_data(param_name, param_range, fixed_params, y0):
    """
    Runs simulations across a range of a single parameter to generate bifurcation data.
    """
    results = {
        'param_values': [],
        'prey_max': [], 'prey_min': [],
        'predator_max': [], 'predator_min': []
    }
    
    # Use tqdm to show a progress bar
    for param_value in tqdm(param_range, desc=f"Simulating for {param_name}"):
        # Update the parameter for this run
        current_params = fixed_params.copy()
        current_params[param_name] = param_value
        
        system = PredatorPreyChemostat(**current_params)
        
        # Run for a long time to ensure we reach a stable state
        t_span = (0, 600)
        t_eval = np.linspace(*t_span, 3000)
        
        sim_data = system.run_simulation(y0, t_span, t_eval)
        
        if not sim_data:
            continue # Skip if simulation failed
            
        # --- Crucial Step: Analyze only the steady-state behavior ---
        # Discard the first half of the simulation to remove transient effects
        steady_state_index = len(sim_data['time']) // 2
        
        steady_prey = sim_data['Chlorella'][steady_state_index:]
        steady_predator = sim_data['Total_Brachionus'][steady_state_index:]
        
        # Store the results
        results['param_values'].append(param_value)
        results['prey_max'].append(np.max(steady_prey))
        results['prey_min'].append(np.min(steady_prey))
        results['predator_max'].append(np.max(steady_predator))
        results['predator_min'].append(np.min(steady_predator))
        
    return results

# --- Setup and Run the Analysis ---
# Define the range for our bifurcation parameter
delta_range = np.linspace(0.1, 1.5, 5000) # 200 points for a high-resolution plot

# Define the other parameters, which will be held constant
fixed_parameters = {'Ni': 80.0, 'lam': 0.4} # Using the paper's standard values
initial_conditions = [60.0, 10.0, 5.0, 5.0]

# Generate the data (this may take a minute or two)
bifurcation_results = generate_bifurcation_data('delta', delta_range, fixed_parameters, initial_conditions)

# # --- Plotting the Bifurcation Diagram ---
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# # Plot for Prey (Chlorella)
# ax1.plot(bifurcation_results['param_values'], bifurcation_results['prey_max'], 'g.', markersize=2, label='Max Prey')
# ax1.plot(bifurcation_results['param_values'], bifurcation_results['prey_min'], 'g.', markersize=2, label='Min Prey')
# ax1.set_ylabel('Prey Population (Chlorella)')
# ax1.set_title('Bifurcation Diagram for Predator-Prey System')
# ax1.grid(True, linestyle='--', alpha=0.6)
# ax1.legend()

# # Plot for Predator (Brachionus)
# ax2.plot(bifurcation_results['param_values'], bifurcation_results['predator_max'], 'k.', markersize=2, label='Max Predator')
# ax2.plot(bifurcation_results['param_values'], bifurcation_results['predator_min'], 'k.', markersize=2, label='Min Predator')
# ax2.set_xlabel('Dilution Rate (δ)')
# ax2.set_ylabel('Predator Population (Brachionus)')
# ax2.grid(True, linestyle='--', alpha=0.6)
# ax2.legend()

# plt.tight_layout()
# plt.show()

# --- REVISED PLOTTING CODE for the Bifurcation Diagram ---

# This code replaces the original plotting section at the end of your script.
# Assumes 'bifurcation_results' dictionary has already been generated.

fig, ax1 = plt.subplots(figsize=(14, 7))

# --- Plot 1: Prey (Chlorella) on the left Y-axis (ax1) ---
color = 'tab:gray' # Using gray for prey as in the paper
ax1.set_xlabel('Dilution rate δ (per day)', fontsize=14)
ax1.set_ylabel('Chlorella vulgaris (Prey)', color=color, fontsize=14)

# To make it look like a solid line, we use a small markersize and no line connecting them
# If your delta_range is dense enough (e.g., 400+ points), this will look continuous.
ax1.plot(bifurcation_results['param_values'], bifurcation_results['prey_max'], '.', color=color, markersize=3)
ax1.plot(bifurcation_results['param_values'], bifurcation_results['prey_min'], '.', color=color, markersize=3)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle=':', alpha=0.7)

# --- Plot 2: Predator (Brachionus) on the right Y-axis (ax2) ---
ax2 = ax1.twinx()  # This creates the second y-axis
color = 'k' # Black for predator
ax2.set_ylabel('Brachionus calyciflorus (Predator)', color=color, fontsize=14)
ax2.plot(bifurcation_results['param_values'], bifurcation_results['predator_max'], '.', color=color, markersize=3, label='Max Predator')
ax2.plot(bifurcation_results['param_values'], bifurcation_results['predator_min'], '.', color=color, markersize=3, label='Min Predator')
ax2.tick_params(axis='y', labelcolor=color)

# Adding a title
plt.title('Bifurcation Diagram (Replication of Fussmann et al., 2000)', fontsize=16)

# Final layout adjustment
fig.tight_layout()
plt.show()