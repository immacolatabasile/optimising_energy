from function import generate_pv_generation, plot_prices_demand
from main import EnergyOptimizationModel

if __name__ == "__main__":
    num_hours = 24
    capacity = 10
    energy_cost = [
        0.15,  # 00:00 - 01:00
        0.15,  # 01:00 - 02:00
        0.15,  # 02:00 - 03:00
        0.15,  # 03:00 - 04:00
        0.15,  # 04:00 - 05:00
        0.15,  # 05:00 - 06:00
        0.20,  # 06:00 - 07:00
        0.20,  # 07:00 - 08:00
        0.20,  # 08:00 - 09:00
        0.20,  # 09:00 - 10:00
        0.20,  # 10:00 - 11:00
        0.20,  # 11:00 - 12:00
        0.25,  # 12:00 - 13:00
        0.25,  # 13:00 - 14:00
        0.15,  # 14:00 - 15:00
        0.15,  # 15:00 - 16:00
        0.20,  # 16:00 - 17:00
        0.30,  # 17:00 - 18:00
        0.30,  # 18:00 - 19:00
        0.30,  # 19:00 - 20:00
        0.20,  # 20:00 - 21:00
        0.20,  # 21:00 - 22:00
        0.20,  # 22:00 - 23:00
        0.15  # 23:00 - 24:00
    ]
    renewable_energy_cost = [
        0.08,  # 00:00 - 01:00
        0.01,  # 01:00 - 02:00
        0.01,  # 02:00 - 03:00
        0.08,  # 03:00 - 04:00
        0.08,  # 04:00 - 05:00
        0.10,  # 05:00 - 06:00
        0.12,  # 06:00 - 07:00
        0.15,  # 07:00 - 08:00
        0.18,  # 08:00 - 09:00
        0.20,  # 09:00 - 10:00
        0.1,  # 10:00 - 11:00
        0.11,  # 11:00 - 12:00
        0.10,  # 12:00 - 13:00
        0.1,  # 13:00 - 14:00
        0.1,  # 14:00 - 15:00
        0.2,  # 15:00 - 16:00
        0.21,  # 16:00 - 17:00
        0.1,  # 17:00 - 18:00
        0.1,  # 18:00 - 19:00
        0.0,  # 19:00 - 20:00
        0.09,  # 20:00 - 21:00
        0.09,  # 21:00 - 22:00
        0.2,  # 22:00 - 23:00
        0.08  # 23:00 - 24:00
    ]

    demand = [
        0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,
        0.18, 0.18, 0.20, 0.17, 0.17, 0.43, 0.43,
        0.18, 0.65, 0.65, 0.25, 0.75, 0.75, 0.75,
        0.28, 0.18, 0.18
    ]
    print("Simulated demand for each hour (kWh):", demand)

    pv_generation = generate_pv_generation(num_hours)
    print("Simulated photovoltaic generation for each hour (Eur):", pv_generation)

    plot_prices_demand(num_hours, energy_cost, renewable_energy_cost, demand, "output/demand.png")

    model = EnergyOptimizationModel(num_hours=num_hours, degradation_costs=0.04,
                                    demand=demand, pv_generation=pv_generation,
                                    energy_cost=energy_cost, renewable_energy_cost=renewable_energy_cost,
                                    max_soc=capacity, capacity=capacity)

    model.solve()

    model.plot_sol("sol.png")
    model.plot_costs("costs.png")
