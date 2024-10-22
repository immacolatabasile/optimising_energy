import matplotlib.pyplot as plt
import numpy as np


def generate_pv_generation(num_hours):
    """Genera una produzione di energia fotovoltaica simulata, con picco nelle ore di sole."""
    # Simuliamo una produzione di energia solare con un picco attorno a mezzogiorno
    base_generation = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.3, 0.3, 0.3, 0.3, 0.3, 0, 0,
                                0, 0, 0, 0, 0, 0, 0])
    return base_generation[:num_hours]


def generate_energy_costs(num_hours):
    base_costs = np.array([np.random.uniform(0.05, 0.2) for _ in range(8)] +  # Ore 0-7
                          [np.random.uniform(0.05, 0.1) for _ in range(8)] +  # Ore 8-15
                          [np.random.uniform(0.2, 0.5) for _ in range(8)])  # Ore 16-23
    return np.round(base_costs[:num_hours], 2)


import numpy as np
import matplotlib.pyplot as plt


def plot_prices_demand(hours, energy_cost_buy, energy_cost_sold, demand, path=None):
    """
    Plots prices and demand over hours.

    :param hours: List of hours (0-23).
    :param prices: List of prices corresponding to the hours.
    :param renewable_energy_cost: List of renewable energy costs corresponding to the hours.
    :param demand: List of demand corresponding to the hours.
    :param path: Path to save the plot image. If None, the plot will be displayed.
    """
    hours = np.arange(0, hours)

    # Check that the lists have the same length
    if len(energy_cost_buy) != len(demand) or len(energy_cost_sold) != len(hours):
        raise ValueError(
            "The lists of hours, energy_cost, renewable_energy_cost, demand, energy_buy, and energy_sold must have the same length.")

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot prices of energy purchase and renewable energy sales
    plt.subplot(3, 1, 1)  # 3 rows, 1 column, first subplot
    plt.plot(hours, energy_cost_buy, marker='o', color='b', label='Energy Bought Price')
    plt.plot(hours, energy_cost_sold, marker='x', color='g', label=' Energy Sold Price')
    plt.title('Energy Bought and Sold per Hour')
    plt.xlabel('Hours')
    plt.ylabel('Price (€/kWh)')
    plt.xticks(hours)
    plt.grid()
    plt.legend()

    # Plot demand
    plt.subplot(3, 1, 2)  # 3 rows, 1 column, second subplot
    plt.plot(hours, demand, marker='o', color='r', label='Demand')
    plt.title('Demand per Hour')
    plt.xlabel('Hours')
    plt.ylabel('Demand (kWh)')
    plt.xticks(hours)
    plt.grid()
    plt.legend()


    plt.xlabel('Hours')
    plt.ylabel('Energy (kWh)')
    plt.xticks(hours)
    plt.grid()
    plt.legend()

    # Save the plot or show it
    plt.tight_layout()
    if path:
        plt.savefig(path)
        print(f"Plot saved as: {path}")
    else:
        plt.show()
