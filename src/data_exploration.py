import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

from citylearn.utilities import get_predictions
from utils import save_multi_image

save = True
ny_data = True

##################################################
# DATA CITYLEARN CHALLENGE 2022
##################################################

weather = pd.read_csv('citylearn/data/test/weather.csv')

#normalize carbon intensity between 0 and 1
carbon_intensity = pd.read_csv('citylearn/data/test/carbon_intensity.csv')['kg_CO2/kWh']
carbon_intensity = (carbon_intensity - np.min(carbon_intensity)) / (np.max(carbon_intensity) - np.min(carbon_intensity))

#normalize prices between 0 and 1
pricing = pd.read_csv('citylearn/data/test/pricing.csv')['Electricity Pricing [$]']
#pricing = (pricing - np.min(pricing)) / (np.max(pricing) - np.min(pricing))

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

fig1, ax1 = plt.subplots()
ax1.plot(xs[:168], pricing[0:168])
ax1.set_title('Electricity Pricing of one week')
ax1.set_ylabel('Electricity Price [$]')
ticks = pd.Series(pd.date_range(start=f'2021-01-01', end=f'2021-01-01 23:00:00', freq='H')).dt.time.astype(str).tolist()*7
ticks = ['Mo']*24
ticks.extend(['Tue']*24)
ticks.extend(['Wed']*24)
ticks.extend(['Thur']*24)
ticks.extend(['Fri']*24)
ticks.extend(['Sat']*24)
ticks.extend(['Sun']*24)
ax1.set_xticks(np.arange(0, len(ticks)))
ax1.xaxis.set_tick_params(length=0)
ax1.set_xticklabels(ticks, rotation=0)
[l.set_visible(False) for (i,l) in enumerate(ax1.get_xticklabels()) if (i-18) % 24 != 0]


##################################################
# REAL DATA FROM NEW YORK ISO AND NREL
##################################################

if ny_data:
    dir = '../datasets/fuel_mix_ny_2021'
    fuel_files = os.listdir(dir)
    fuel = pd.read_csv(f'{dir}/{fuel_files[0]}')
    fuel = (fuel.pivot_table(index=['Time Stamp'],
                             columns='Fuel Category',
                             values='Gen MW').reset_index().rename_axis(None, axis=1))

    for file in fuel_files[1:]:
        new = pd.read_csv(f'{dir}/{file}')
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


##################################################
# PREPROCESSED FUEL MIX DATA
##################################################
fuel_mix = pd.read_csv('citylearn/data/nydata/fuelmix.csv')

sum = np.array(fuel_mix['Renewable Sources [kWh]']+fuel_mix['Other [kWh]'])
fig, ax = plt.subplots()
ax.plot(np.arange(sum.size), sum)
ax.set_title('Total Energy produced')
ax.set_ylabel('Energy produced [$MW$]')
ax.set_xlabel('Time step')

percent = np.array(fuel_mix['Renewable Sources [kWh]']/sum)*100
fig, ax = plt.subplots()
ax.plot(np.arange(percent.size), percent)
ax.set_title('$\%$ Renweable Energy from total produced Energy')
ax.set_ylabel('%')
ax.set_xlabel('Time step')

print(f'Min Renewable %: {percent.min()}')
print(f'Max Renewable %: {percent.max()}')

##################################################
# PREPROCESSED NY WEATHER DATA
##################################################
ny_weather_prep = pd.read_csv('citylearn/data/nydata/weather.csv')
dhi = np.array(ny_weather_prep['Diffuse Solar Radiation [W/m2]'])
dni = np.array(ny_weather_prep['Direct Solar Radiation [W/m2]'])

# Correlation DHI - Solar generation B1
solar_generation = np.array(pd.read_csv('citylearn/data/nydata/Building_1.csv')['Solar Generation [W/kW]'])
fig, ax = plt.subplots()
ax.scatter(dhi, solar_generation, s=1)
ax.set_title('Solar generation Building 1 vs. DHI (NY data)')
ax.set_ylabel('Solar generation [$W/kW$]')
ax.set_xlabel('DHI [$W/m^2$]')

fig, ax = plt.subplots()
ax.scatter(dni, solar_generation, s=1)
ax.set_title('Solar generation Building 1 vs. DNI (NY data)')
ax.set_ylabel('Solar generation [$W/kW$]')
ax.set_xlabel('DNI [$W/m^2$]')

# Correlation DHI - Solar generation B6
solar_generation = np.array(pd.read_csv('citylearn/data/nydata/Building_6.csv')['Solar Generation [W/kW]'])
fig, ax = plt.subplots()
ax.scatter(dhi, solar_generation, s=1)
ax.set_title('Solar generation Building 6 vs. DHI (NY data)')
ax.set_ylabel('Solar generation [$W/kW$]')
ax.set_xlabel('DHI [$W/m^2$]')

fig, ax = plt.subplots()
ax.scatter(dni, solar_generation, s=1)
ax.set_title('Solar generation Building 6 vs. DNI (NY data)')
ax.set_ylabel('Solar generation [$W/kW$]')
ax.set_xlabel('DNI [$W/m^2$]')

# Correlation DHI - Solar generation B14
solar_generation = np.array(pd.read_csv('citylearn/data/nydata/Building_14.csv')['Solar Generation [W/kW]'])
fig, ax = plt.subplots()
ax.scatter(dhi, solar_generation, s=1)
ax.set_title('Solar generation Building 14 vs. DHI (NY data)')
ax.set_ylabel('Solar generation [$W/kW$]')
ax.set_xlabel('DHI [$W/m^2$]')

fig, ax = plt.subplots()
ax.scatter(dni, solar_generation, s=1)
ax.set_title('Solar generation Building 14 vs. DNI (NY data)')
ax.set_ylabel('Solar generation [$W/kW$]')
ax.set_xlabel('DNI [$W/m^2$]')

##################################################
# PRICING DATA
##################################################

pricing_new = pd.read_csv('citylearn/data/nydata/pricing.csv')['Electricity Pricing [$]']

fossil_share = 1-fuel_mix['Renewable Share']
alpha = 10
pricing_new = pricing_new + alpha * fossil_share

# set minimum price to 0.2
pricing_new = pricing_new - pricing_new.min() + 0.2

fig, ax = plt.subplots()
ax.plot(np.arange(len(pricing_new)), pricing_new)
ax.set_title(f'Electricity Pricing [$] - Weighted by Fossil Energy share (factor = {alpha})')
ax.set_ylabel('$')
ax.set_xlabel('Time step')

fig, ax = plt.subplots()
ax.plot(np.arange(168), pricing_new[:168], label='Prices influenced by fossil energy')
ax.plot(np.arange(168), pricing[:168], label='Old prices')
ax.set_title(f'Electricity Pricing [$] - Original vs. Weighted by Fossil Energy share (factor = {alpha})')
ax.set_ylabel('$')
ax.set_xticks(np.arange(0, len(ticks)))
ax.xaxis.set_tick_params(length=0)
ax.set_xticklabels(ticks, rotation=0)
[l.set_visible(False) for (i,l) in enumerate(ax.get_xticklabels()) if (i-18) % 24 != 0]

if save:
    filename = "../datasets/data_exploration_plots/exploration-plots_" + datetime.now().strftime("%Y%m%dT%H%M%S")
    save_multi_image(filename)
else:
    plt.show()