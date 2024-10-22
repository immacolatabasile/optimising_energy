import sys

import numpy as np


from engine import EnergyOptimizationModel
from function import generate_energy_costs, generate_pv_generation, plot_prices_demand

if __name__ == "__main__":
    num_hours = 24
    capacity = 500
    energy_cost = generate_energy_costs(num_hours)
    print("Costi dell'energia per ogni ora (Eur):", energy_cost)

    demand = np.random.uniform(200, 400, size=num_hours)
    demand = np.round(demand, 2)
    print("Domanda simulata per ogni ora (Eur):", demand)

    pv_generation = generate_pv_generation(num_hours)
    print("Produzione fotovoltaica simulata per ogni ora (Eur):", pv_generation)

    plot_prices_demand(num_hours, energy_cost, demand)

    model = EnergyOptimizationModel(num_hours=num_hours, degradation_costs=0.04,
                                    demand=demand, pv_generation=pv_generation,
                                    energy_cost=energy_cost, max_soc=capacity, capacity=capacity)

    # Risoluzione del modello
    model.solve()

    model.plot_sol()
    # Plottaggio dei risultati
    model.plot_costs()
