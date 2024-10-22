import numpy as np
from matplotlib import pyplot as plt
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpStatus, value

from function import generate_pv_generation, generate_energy_costs, plot_prices_demand


class EnergyOptimizationModel:
    def __init__(self, num_hours=24, degradation_costs=150, demand=None,
                 pv_generation=None, energy_cost=None, renewable_energy_cost=None, max_soc=100, capacity=300):
        self.max_soc = max_soc
        self.capacity = capacity
        self.initial_soc = 0
        self.energy_cost = energy_cost if energy_cost is not None else np.random.uniform(0.05, 0.2, size=num_hours)
        self.renewable_energy_cost = renewable_energy_cost if renewable_energy_cost is not None else np.random.uniform(
            0.05, 0.1, size=num_hours)
        if demand is None:
            self.demand = np.random.uniform(200, 400, size=num_hours).tolist()
        else:
            self.demand = demand

        if pv_generation is None:
            self.pv_generation = np.random.uniform(0, 300, size=num_hours).tolist()
        else:
            self.pv_generation = pv_generation
        self.num_hours = num_hours
        self.degradation_costs = degradation_costs
        self.energy_buy = LpVariable.dicts("E_buy", range(num_hours), lowBound=0)
        self.energy_sold = LpVariable.dicts("E_sold", range(num_hours), lowBound=0)
        self.battery_input = LpVariable.dicts("E_inp", range(num_hours), lowBound=0, upBound=self.capacity)
        self.battery_output = LpVariable.dicts("E_out", range(num_hours), lowBound=0, upBound=self.capacity)
        self.y = LpVariable.dicts("y", range(num_hours), cat='Binary')
        self.s = LpVariable.dicts("s", range(num_hours), cat='Binary')
        self.soc = LpVariable.dicts("SOC", range(num_hours), lowBound=0)
        self.is_positive = LpVariable.dicts(f"is_positive", range(num_hours), cat='binary')
        # Example fixed values for the simulation
        self.demand = demand
        self.pv_generation = pv_generation

        # Create the optimization problem
        self.problem = LpProblem("Energy_Optimization", LpMinimize)

    def export_constraints_to_file(self, filename):
        with open(filename, 'w') as file:
            file.write("Objective Function:\n")
            file.write(str(self.problem.objective) + '\n\n')

            file.write("Constraints:\n")
            for name, constraint in self.problem.constraints.items():
                file.write(f"{name}: {constraint}\n")
        print(f"Constraints exported to {filename}")

    def setup_problem(self):
        # Objective function: minimize operational costs + degradation costs
        global t
        self.problem += lpSum(
            self.energy_cost[t] * self.energy_buy[t] - self.renewable_energy_cost[t] * self.energy_sold[t] for t in
            range(self.num_hours)) + \
                        self.degradation_costs * (
                            lpSum(self.battery_input[t] + self.battery_output[t] for t in range(self.num_hours))), \
            "Total_Cost"
        # Energy balance constraint
        for t in range(self.num_hours):
            self.problem += (self.demand[t] == self.pv_generation[t] +
                             self.energy_buy[t] - self.energy_sold[t] - self.battery_input[t] + self.battery_output[t])

        for t in range(self.num_hours):
            self.problem += (self.soc[t] <= self.max_soc)
            if t == 0:
                # SOC (initial time)
                self.problem += (
                    self.soc[t] == self.initial_soc + self.battery_input[t] - self.battery_output[t],
                    f"SOC_balance_{t}")
            else:
                self.problem += (
                    self.soc[t] == self.soc[t - 1] + self.battery_input[t] - self.battery_output[t],
                    f"SOC_balance_{t}")
        # Add the constraints for the binary variable
        M = self.capacity  # A large number, you can adjust this based on your needs

        for t in range(self.num_hours):
            self.problem += (self.battery_input[t] <= self.y[t] * M, f"E_inp_constraint_{t}")
            self.problem += (self.battery_output[t] <= (1 - self.y[t]) * M, f"E_out_constraint_{t}")
            self.problem += (self.energy_buy[t] <= self.s[t] * M, f"E_buy_constraint_{t}")
            self.problem += (self.energy_sold[t] <= (1 - self.s[t]) * M, f"E_sold_constraint_{t}")

            self.problem += self.battery_input[t] <= self.pv_generation[t], f"E_buy_constraint_prod_{t}"

            self.problem += (self.pv_generation[t] - self.demand[t] >= 0) == self.is_positive[t]

            self.problem += (
                self.battery_input[t] <= self.pv_generation[t] - self.demand[t] + M * (1 - self.is_positive[t]),
                f"E_bu1y_constraint_prod_{t}")

    def solve(self):
        self.setup_problem()
        self.export_constraints_to_file("vincoli.csv")
        self.problem.solve()
        print(f"Status: {LpStatus[self.problem.status]}")
        print(f"Solver used: {self.problem.solver}")

        for t in range(self.num_hours):
            print(f"Hour {t}: "
                  f"E_buy = {value(self.energy_buy[t])},"
                  f"E_sold = {value(self.energy_sold[t])},"
                  f" E_inp = {value(self.battery_input[t])}, "
                  f"E_out = {value(self.battery_output[t])},"
                  f"Y = {value(self.y[t])},"
                  f" SOC = {value(self.soc[t])}")

    def plot_costs(self, path=None):
        energy_buy = [value(self.energy_buy[t]) for t in range(self.num_hours)]
        demand = self.demand
        pv_generation = self.pv_generation
        battery_input = [value(self.battery_input[t]) for t in range(self.num_hours)]
        battery_output = [value(self.battery_output[t]) for t in range(self.num_hours)]

        plt.figure(figsize=(12, 8))

        # Plotta i costi operativi con sfumatura
        plt.fill_between(range(self.num_hours), energy_buy, color='blue', alpha=0.3,
                         label='Energy buy (KW)')
        plt.plot(range(self.num_hours), energy_buy, marker='o', linestyle='-', color='b')

        # Plotta la domanda
        plt.plot(range(self.num_hours), demand, marker='x', linestyle='--', color='red', label='Demand (KW)')

        # Plotta la produzione fotovoltaica con sfumatura
        plt.fill_between(range(self.num_hours), pv_generation, color='green', alpha=0.3, label='Produzione PV (KW)')
        plt.plot(range(self.num_hours), pv_generation, marker='s', linestyle=':', color='g')

        # Plotta l'energia in entrata dalla batteria con sfumatura
        plt.fill_between(range(self.num_hours), battery_input, color='orange', alpha=0.3,
                         label='Energia in entrata (KW)')
        plt.plot(range(self.num_hours), battery_input, marker='^', linestyle='-', color='orange')

        # Plotta l'energia in uscita dalla batteria con sfumatura
        plt.fill_between(range(self.num_hours), battery_output, color='purple', alpha=0.3,
                         label='Energia in uscita (KW)')
        plt.plot(range(self.num_hours), battery_output, marker='v', linestyle='-', color='purple')
        # Aumenta i costi dell'energia per renderli più visibili
        scaled_energy_cost = [cost for cost in self.energy_cost]  # Moltiplica per 10 o altro fattore
        plt.plot(range(self.num_hours), scaled_energy_cost, marker='D', linestyle='-', color='black',
                 label='Costi dell\'energia (Eur)', linewidth=2)

        # total_energy = [energy_buy[t] + pv_generation[t] for t in range(self.num_hours)]
        # Plotta la somma di energia acquistata e prodotta
        # plt.fill_between(range(self.num_hours), total_energy, color='cyan', alpha=0.3,
        #                  label='Energia Totale (Comprata + Prodotta) ')
        # plt.plot(range(self.num_hours), total_energy, marker='D', linestyle='-', color='cyan')

        # Aggiungi titoli e etichette
        plt.title('Variazione dei costi operativi, demand e produzione nelle 24 ore')
        plt.xlabel('hour')
        plt.ylabel('Value (Eur)')
        plt.xticks(range(self.num_hours))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if path:
            plt.savefig(path)
            print(f"Plot saved as: {path}")
        else:
            plt.show()

    def plot_sol(self, path=None):
        num_hours = self.num_hours
        fig, ax = plt.subplots(figsize=(12, 8))
        energy_buy = [value(self.energy_buy[t]) for t in range(self.num_hours)]
        energy_sold = [value(self.energy_sold[t]) for t in range(self.num_hours)]
        total_soc = [value(self.soc[t]) for t in range(self.num_hours)]

        bar_width = 0.25
        indices = np.arange(num_hours)

        bars0 = ax.bar(indices, self.pv_generation, width=bar_width, label='Energy (KW)', color='#1f77b4')  # Blue
        bars1 = ax.bar(indices, self.demand, width=bar_width, label='Demand Energy (KW)', color='#ff7f0e')  # Orange
        bars2 = ax.bar(indices + bar_width, energy_buy, width=bar_width, label='Purchased Energy (KW)',
                       color='#2ca02c')  # Green
        bars3 = ax.bar(indices + bar_width, energy_sold, width=bar_width, label='Sold Energy (KW)',
                       color='#d62728')  # Red
        bars4 = ax.bar(indices + 2 * bar_width, total_soc, width=bar_width, label='SOC (%)', color='#9467bd')  # Purple

        ax.set_xlabel('Hours of the Day')
        ax.set_ylabel('Value (KW / %)')
        ax.set_title('Energy Production and Consumption by Hour')
        ax.set_xticks(indices + bar_width)
        ax.set_xticklabels([f'Hour {i}' for i in range(num_hours)])
        ax.legend()
        ax.grid(axis='y')

        for bar in bars0 + bars1 + bars2 + bars3 + bars4:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 punti di offset in alto
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.tight_layout()
        if path:
            plt.savefig(path)
            print(f"Plot saved as: {path}")
        else:
            plt.show()
