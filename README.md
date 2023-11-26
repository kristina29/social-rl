# Master's Thesis Project: Social Learning in Multi-Agent Reinforcement Learning for Carbon Emission Reduction

## Overview

This repository hosts the implementation of my master's thesis conducted for the Master of Science degree in Computer Science at the University of Tübingen. The thesis focuses on 'Incorporating Social Learning into Multi-Agent Reinforcement Learning to Lower Carbon Emissions in Energy Systems'. The primary goal is to explore the impact of social learning methods on the efficiency of energy systems, with a significant emphasis on reducing carbon emissions.

## Thesis Objectives

- To integrate social learning methods, specifically imitation learning and decision biasing, into the Soft Actor-Critic (SAC) algorithm.
- To enhance the Multi-Agent Reinforcement Learning with Imitation and Social Awareness (MARLISA) algorithm for better performance regarding carbon emissions.
- To assess the algorithms' performance based on the KPIs of *fossil energy consumption*, *average fossil share*, *average fossil share grid*, and *1 - used PV* by the agents.
- To compare the performance of the social methods to asocial baseline SAC.

## Framework Used

The project is built upon the [CityLearn environment](https://www.citylearn.net), which has been customized to suit the specific requirements of this research. CityLearn provides a rich simulation environment for demand response and urban energy management tasks, facilitating the study of the proposed algorithms' effectiveness.

## Repository Structure

- `src/`: Contains all the source code developed for the thesis and the CityLearn code.
  - `src/citylearn/agents`: Contains all the source code of the RL agents (provided by CityLearn and developed for the thesis).
  - `src/citylearn/data`: Contains the datasets used in the CityLearn simulations.
  - `src/scripts`: Contains the python scripts for data exploration, data preprocessing and generating plots. 
- `datasets/`: Includes datasets and the results from simulations.
- `experiments/`:  Includes configuration details and results of all experiments (Some of them may be based on old options).
- `thesis/`: LaTeX source code for the thesis document.

## Getting Started

To get started with the project:

- Clone the repository.
- Install the required dependencies listed in `requirements.txt` or create a conda environment using `env.yml` and activate it.
- Use one of the following commands to run a social simulation (**Attention! With standard parameters, the running time is around 45 minutes**):
  - Train RBC, asocial SAC and SAC agents using imitation learning for two buildings using the optimized hyperparameters:
    ```
    python3 src/socialrl.py -s nnb_limitobs1 -b 2 --rbc --sac --transitions datasets/transitions/SAC_transitions_b5.pkl --autotune
    ``` 
    [Runtime ~17 Minutes]
  - Train RBC and SAC-DemoPol agents for two buildings using the optimized hyperparameters, two random demonstrators, mode 1, and an imitation learning rate $\alpha_i=0.2$:
    ```
    python3 src/socialrl.py -s nnb_limitobs1 -b 2 -d 2 --rbc --sacdemopol --autotune --mode 1 --ir 0.2
    ```
    [Runtime ~x Minutes]
  - Train RBC and SAC-DemoQ agents for two buildings using the optimized hyperparameters, two random demonstrators, and an imitation learning rate $\alpha_i=0.15$:
    ```
    python3 src/socialrl.py -s nnb_limitobs1 -b 2 -d 2 --rbc --sacdemoq --autotune --ir 0.15
    ```
    [Runtime ~x Minutes]
  - Train RBC and MARLISA agents for two buildings using the optimized hyperparameters and information sharing. The reward function to use is defined in `src/citylearn/data/nnb_limitobs1_marlisa/schema.json`:
    ```
    srun python3 src/marlisa_social.py -s nnb_limitobs1_marlisa -b 2 --autotune --information_sharing
    ```
    [Runtime ~x Minutes]
  
For enhanced flexibility and customization, you have the ability to fine-tune the simulations by configuring additional options. This includes using a pre-trained demonstrator to guide the agents or adjusting various hyperparameters. To explore the full range of configurable settings for both non-social and social simulations, please refer to `src/options.py` or run `python3 src/socialrl.py --help`.
