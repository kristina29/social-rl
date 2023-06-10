import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from utils import save_multi_image

save = True

##################################################
# DATA CITYLEARN CHALLENGE 2022
##################################################

weather = pd.read_csv('citylearn/data/test/weather.csv')

#normalize carbon intensity between 0 and 1
carbon_intensity = pd.read_csv('citylearn/data/test/carbon_intensity.csv')['kg_CO2/kWh']
carbon_intensity = (carbon_intensity - np.min(carbon_intensity)) / (np.max(carbon_intensity) - np.min(carbon_intensity))

#normalize direct solar radiation between 0 and 1
direct_solar = np.array(weather['Direct Solar Radiation [W/m2]'])
direct_solar = (direct_solar - np.min(direct_solar)) / (np.max(direct_solar) - np.min(direct_solar))

#normalize diffuse solar radiation between 0 and 1
diffuse_solar = np.array(weather['Diffuse Solar Radiation [W/m2]'])
diffuse_solar = (direct_solar - np.min(direct_solar)) / (np.max(direct_solar) - np.min(direct_solar))

xs = np.arange(0, len(carbon_intensity))

fig1, ax1 = plt.subplots()
ax1.plot(xs, direct_solar, label='Direct solar radiation [$W/M^2$]')
ax1.plot(xs, carbon_intensity, label='Carbon intensity [$kg_CO2/kWh$]')
ax1.legend()
ax1.set_title('Carbon intensity vs. direct solar radiation')
ax1.set_xlabel('Time step')
ax1.set_ylabel('Values')

fig2, ax2 = plt.subplots()
ax2.scatter(carbon_intensity, direct_solar, s=1)
ax2.set_title('Carbon intensity vs. direct solar radiation')
ax2.set_xlabel('Carbon intensity [$kg_CO2/kWh$]')
ax2.set_ylabel('Direct solar radiation [$W/M^2$]')

fig5, ax5 = plt.subplots()
ax5.scatter(direct_solar, diffuse_solar, s=1)
ax5.set_title('Diffuse solar radiation vs. direct solar radiation')
ax5.set_xlabel('Diffuse solar radiation [$W/M^2$]')
ax5.set_ylabel('Direct solar radiation [$W/M^2$]')


##################################################
# REAL DATA FROM NEW YORK ISO AND NREL
##################################################

fuel_files = os.listdir('../datasets/fuel_mix_ny')
fuel = pd.read_csv('../datasets/fuel_mix_ny/' + fuel_files[0])
fuel = (fuel.pivot_table(index=['Time Stamp'],
                         columns='Fuel Category',
                         values='Gen MW').reset_index().rename_axis(None, axis=1))

for file in fuel_files[1:]:
    new = pd.read_csv('../datasets/fuel_mix_ny/' + file)
    new = (new.pivot_table(index=['Time Stamp'],
                           columns='Fuel Category',
                           values='Gen MW').reset_index().rename_axis(None, axis=1))
    fuel = fuel.append(new)
fuel['Time Stamp'] = pd.to_datetime(fuel['Time Stamp'], format='%m/%d/%Y %H:%M:%S')

ny_weather = pd.read_csv('../datasets/weather_ny_42.30_-74.37_2021.csv', skiprows=2)
ny_weather['Time Stamp'] = pd.to_datetime(ny_weather[['Month', 'Day', 'Year', 'Hour', 'Minute']]
                                          .astype(str).apply(' '.join, 1), format='%m %d %Y %H %M')
ny_weather = ny_weather.drop(columns=['Month', 'Day', 'Year', 'Hour', 'Minute'])

joined = pd.merge(fuel, ny_weather, on='Time Stamp', sort=True)

fig, ax = plt.subplots()
ax.scatter(joined['Wind Speed'], joined['Wind'], s=1)
ax.set_title('Wind energy vs. wind speed')
ax.set_ylabel('Wind energy [$MW$]')
ax.set_xlabel('Wind speed [$m/s$]')

fig, ax = plt.subplots()
ax.scatter(joined['DHI'], joined['Other Renewables'], s=1)
ax.set_title('Other renewables energy vs. DHI')
ax.set_ylabel('Other renewables energy [$MW$]')
ax.set_xlabel('DHI [$W/m^2$]')

fig, ax = plt.subplots()
ax.scatter(joined['DNI'],joined['Other Renewables'], s=1)
ax.set_title('Other renewables energy vs. DNI')
ax.set_ylabel('Other renewables energy [$MW$]')
ax.set_xlabel('DNI [$W/m^2$]')

if save:
    filename = "../datasets/data_exploration_plots/exploration-plots_" + datetime.now().strftime("%Y%m%dT%H%M%S")
    save_multi_image(filename)
else:
    plt.show()