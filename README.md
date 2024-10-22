# Energy Optimization Model

## Overview

The **Energy Optimization Model** is a Python-based simulation tool designed to optimise energy consumption and production for a renewable energy system over a 24-hour period. This model integrates various parameters, including energy costs, demand, photovoltaic (PV) generation, and battery management to minimise operational and degradation costs.

## Features

- **Dynamic Energy Cost Generation**: Generates energy costs randomly within a specified range to simulate real-world scenarios.
- **PV Generation Simulation**: Simulates PV generation for each hour, which can be customised or randomly generated.
- **Battery Management**: Implements a battery management system to optimise energy storage and usage.
- **Cost Analysis**: Calculates operational costs, degradation costs, and provides visual outputs for better understanding.
- **Graphical Output**: Produces visual plots of energy production, consumption, and operational costs for analysis.

## Requirements

To run the Energy Optimization Model, ensure you have the following packages installed:

- Python 3.x
- NumPy
- Matplotlib
- PuLP

You can install the required packages using pip:
    
    pip install numpy matplotlib pulp

## Usage

To use the Energy Optimization Model, follow these steps:

1. **Setup the Model**: Initialize the model by specifying the number of hours (e.g., 24) and the degradation costs.
   
   ```python
   model = EnergyOptimizationModel(num_hours=24, degradation_costs=150)

## SETUP
    model = solve()

## Export Constraints
    model.export_constraints_to_file("constraints.csv")

## Plotting Results:
Visualise the results using the provided plotting methods:

    model.plot_costs()
    model.plot_sol()

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
Thanks to the contributors and libraries that made this project possible.
Special thanks to the community for feedback and suggestions.



