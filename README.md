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

- **src/**: Contains all the source code developed for the thesis and the CityLearn code.
- **datasets/**: Includes datasets and the results from simulations.
- **experiments/**: 
- **thesis/**:

## Key Results

A detailed analysis of the fossil energy consumption by the agents using the adapted CityLearn framework.
Comparative performance metrics and findings from the implementation of social learning strategies in multi-agent reinforcement learning.

## Getting Started

To get started with the project:

- Clone the repository.
- Install the required dependencies listed in requirements.txt.
- Follow the setup instructions in the installation.md to configure your environment.
- Navigate to the examples/ directory to run sample simulations.


## Contact Information

For any queries regarding the project, please contact [Your Name] at [Your Email].