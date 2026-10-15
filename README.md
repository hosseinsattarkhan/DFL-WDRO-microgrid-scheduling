# Decision-Focused Learning for Wasserstein DRO Microgrid Scheduling

Python implementation of the framework described in V23 of the manuscript.

## Architecture

Two parallel implementations:

- **models/pyomo/** -- Pyomo + HiGHS (main results, no license required)
- **models/gams/** -- GAMS + HiGHS/CPLEX (comparison)

## Installation

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

## Data

- NREL NSRDB -- solar irradiance
- NREL WIND Toolkit -- wind speed
- MISO Day-Ahead Market -- prices

## Structure

- config/ -- YAML configuration
- models/pyomo/ -- Pyomo models
- models/gams/ -- GAMS models
- src/ -- Python source
- experiments/ -- Experiment scripts
- results/ -- Outputs
