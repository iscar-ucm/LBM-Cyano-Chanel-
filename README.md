# LBM fluid + cellDEVS cyanobacteria simulation
This is a repository containing the code referenced in the paper "Simulation of cyanobacteria behavior in water bodies with cell-DEVS and Lattice-Boltzmann methods" presented in ANNSIM'25 by Ferrero-Losada et al.

Here are contained both the fluid LBM simulation and the cellular automata simulation of cell-DEVS. However, to run the latter, xDEVS v3.0.0 must be installed first from its own repository.

A brief explanation on the contents of each file:
 - *Lattice Boltzmann 3D.py*: LBM simulation code. The resulting data is saved in a folder with a file per saved timestep.
   
 - *flux_process.py*: Python code to process the LBM data into cell-DEVS interface flow data.
   
 - *flowdata.pkl*: This file contains the cell-DEVS cells interface flow data. It is the output of *flux_process.py*
   
      ![particion2D](https://github.com/user-attachments/assets/51e59166-5515-4948-b73b-a5929a4878de)
 - *main.py*: Main cell-DEVS simulation file. Here the simulated span of time is defined.

 - *sir_cell.py*: Here the case specific models of the internal computations carried out by each cell module of the cellular automata are defined.
The local parameters of the cell configuration, models and state are defined. Upon inspection of the *cell.py* file of cell-DEVS, the local computations are carried out in the external delta of eac cell.

 - *scenario.json*: In this file the cell mesh size, initial parameter values, state and configuration of each cell are stablished. Vicinity relations are also defined.
   
 - *sir_sink.py*: This is a (currently) unused file that could be used in the future to calculate global parameters of the water body. It has no role inside the ANNSIM'25 paper.
   
 - *sir_coupled.py*: In this file *sir_sink.py* is coupled with the rest of the scenario cells. It has no role inside the ANNSIM'25 paper.
