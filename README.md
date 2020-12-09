# Implementing Trust Region Policy Optimisation for Deep Reinforcement Learning
Nikolas Wilhelm


## Algorithms

This project contains Implementations of the following algorithms,
provided in "RL_ALGORITHMS":

1. [REINFORCE](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)
2. [A2C](https://arxiv.org/abs/1602.01783)
3. [TNPG](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/NaturalActorCritic.pdf)
4. [TRPO](https://arxiv.org/abs/1502.05477)
5. [PPO](https://blog.openai.com/openai-baselines-ppo/)
6. [PPO-soft](https://drive.google.com/file/d/1V4pD6HHCLowd_OdXfO4NddPGNb6baV-y/view)
7. [DDPG](https://arxiv.org/abs/1509.02971)
8. [SAC](https://arxiv.org/abs/1801.01290)
9. [TD3](https://arxiv.org/pdf/1802.09477.pdf#cite.popov2017data)


## Environments

Further following ENVIRONMENTS are provided:

1. LinearModel
2. MountainCar
3. InvPendulum
4. Segway
5. BB8

## Setup

To setup all modules in [Julia](https://julialang.org/downloads/), please follow the instructions in the
"./startup.jl" file provided in this repository.

## Run Simulations

To train a controller on a desired system:
1. Modify the "./RL_ALGORITHMS/src/init.jl" file with the desired requirements.
2. Execute the file corresponding to the desired structure.

An example simulation is thereby executed by navigating to the "Master_TRPO_Niko" folder and typing:
```
julia
include("RL_Algorithms/ppo_soft.jl")  #To execute a specific algorithm, here ppo-soft
include("RL_Algorithms/benchmark_algorithms.jl") #To perform a benchmark with all algorithms
```


## Results
Please visit: [RESULTS](https://sites.google.com/view/implenenting-trpo-and-ppo/overview)
