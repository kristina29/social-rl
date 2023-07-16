import glob
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

from mpl_toolkits.basemap import Basemap

from utils import save_multi_image

save = True
challenge_data = True
ny_data = True

##################################################
# DATA CITYLEARN CHALLENGE 2022
##################################################

if challenge_data:
    weather = pd.read_csv('citylearn/data/test/weather.csv')

    # normalize carbon intensity between 0 and 1
    carbon_intensity = pd.read_csv('citylearn/data/test/carbon_intensity.csv')['kg_CO2/kWh']
    carbon_intensity = (carbon_intensity - np.min(carbon_intensity)) / (
                np.max(carbon_intensity) - np.min(carbon_intensity))

    # normalize prices between 0 and 1
    pricing = pd.read_csv('citylearn/data/test/pricing.csv')['Electricity Pricing [$]']
    # pricing = (pricing - np.min(pricing)) / (np.max(pricing) - np.min(pricing))

    # normalize direct solar radiation between 0 and 1
    direct_solar = np.array(weather['Direct Solar Radiation [W/m2]'])
    direct_solar = (direct_solar - np.min(direct_solar)) / (np.max(direct_solar) - np.min(direct_solar))

    # normalize diffuse solar radiation between 0 and 1
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
    ticks = pd.Series(pd.date_range(start=f'2021-01-01', end=f'2021-01-01 23:00:00', freq='H')).dt.time.astype(
        str).tolist() * 7
    ticks = ['Mo'] * 24
    ticks.extend(['Tue'] * 24)
    ticks.extend(['Wed'] * 24)
    ticks.extend(['Thur'] * 24)
    ticks.extend(['Fri'] * 24)
    ticks.extend(['Sat'] * 24)
    ticks.extend(['Sun'] * 24)
    ax1.set_xticks(np.arange(0, len(ticks)))
    ax1.xaxis.set_tick_params(length=0)
    ax1.set_xticklabels(ticks, rotation=0)
    [l.set_visible(False) for (i, l) in enumerate(ax1.get_xticklabels()) if (i - 18) % 24 != 0]

    if save:
        filename = "../datasets/data_exploration_plots/citylearn-challenge-2022-plots_" + datetime.now().strftime(
            "%Y%m%dT%H%M%S")
        save_multi_image(filename)
    else:
        plt.show()

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
    ax.scatter(joined['DNI'], joined['Other Renewables'], s=1)
    ax.set_title('Other renewables energy vs. DNI')
    ax.set_ylabel('Other renewables energy [$MW$]')
    ax.set_xlabel('DNI [$W/m^2$]')

    # include new weather files
    ny_weather2 = pd.read_csv('../datasets/weather_ny_40.86_-72.57_2021.csv', skiprows=2)
    ny_weather2['Time Stamp'] = pd.to_datetime(
        ny_weather2[['Month', 'Day', 'Year', 'Hour', 'Minute']].astype(str).apply(' '.join, 1), format='%m %d %Y %H %M')
    ny_weather2 = ny_weather2.drop(columns=['Month', 'Day', 'Year', 'Hour', 'Minute'])
    ny_weather3 = pd.read_csv('../datasets/weather_ny_42.44_-78.45_2021.csv', skiprows=2)
    ny_weather3['Time Stamp'] = pd.to_datetime(
        ny_weather3[['Month', 'Day', 'Year', 'Hour', 'Minute']].astype(str).apply(' '.join, 1), format='%m %d %Y %H %M')
    ny_weather3 = ny_weather3.drop(columns=['Month', 'Day', 'Year', 'Hour', 'Minute'])
    ny_weather4 = pd.read_csv('../datasets/weather_ny_44.86_-73.61_2021.csv', skiprows=2)
    ny_weather4['Time Stamp'] = pd.to_datetime(
        ny_weather4[['Month', 'Day', 'Year', 'Hour', 'Minute']].astype(str).apply(' '.join, 1), format='%m %d %Y %H %M')
    ny_weather4 = ny_weather4.drop(columns=['Month', 'Day', 'Year', 'Hour', 'Minute'])
    ny_weather5 = pd.read_csv('../datasets/weather_ny_41.14_-73.91_2021.csv', skiprows=2)
    ny_weather5['Time Stamp'] = pd.to_datetime(
        ny_weather5[['Month', 'Day', 'Year', 'Hour', 'Minute']].astype(str).apply(' '.join, 1), format='%m %d %Y %H %M')
    ny_weather5 = ny_weather5.drop(columns=['Month', 'Day', 'Year', 'Hour', 'Minute'])
    ny_weather6 = pd.read_csv('../datasets/weather_ny_43.10_-75.33_2021.csv', skiprows=2)
    ny_weather6['Time Stamp'] = pd.to_datetime(
        ny_weather6[['Month', 'Day', 'Year', 'Hour', 'Minute']].astype(str).apply(' '.join, 1), format='%m %d %Y %H %M')
    ny_weather6 = ny_weather6.drop(columns=['Month', 'Day', 'Year', 'Hour', 'Minute'])
    ny_weather7 = pd.read_csv('../datasets/weather_ny_43.14_-73.97_2021.csv', skiprows=2)
    ny_weather7['Time Stamp'] = pd.to_datetime(
        ny_weather7[['Month', 'Day', 'Year', 'Hour', 'Minute']].astype(str).apply(' '.join, 1), format='%m %d %Y %H %M')
    ny_weather7 = ny_weather7.drop(columns=['Month', 'Day', 'Year', 'Hour', 'Minute'])
    ny_weather8 = pd.read_csv('../datasets/weather_ny_43.38_-76.31_2021.csv', skiprows=2)
    ny_weather8['Time Stamp'] = pd.to_datetime(
        ny_weather8[['Month', 'Day', 'Year', 'Hour', 'Minute']].astype(str).apply(' '.join, 1), format='%m %d %Y %H %M')
    ny_weather8 = ny_weather8.drop(columns=['Month', 'Day', 'Year', 'Hour', 'Minute'])

    ny_weather_mean = pd.concat([ny_weather, ny_weather2, ny_weather3, ny_weather4, ny_weather5, ny_weather6,
                                 ny_weather7, ny_weather8]).groupby('Time Stamp', as_index=False).median()

    joined = pd.merge(fuel, ny_weather_mean, on='Time Stamp', sort=True)

    fig, ax = plt.subplots()
    ax.scatter(joined['Wind Speed'], joined['Wind'], s=1)
    ax.set_title('Wind energy vs. wind speed (8 measure locs median)')
    ax.set_ylabel('Wind energy [$MW$]')
    ax.set_xlabel('Wind speed [$m/s$]')

    fig, ax = plt.subplots()
    ax.scatter(joined['DHI'], joined['Other Renewables'], s=1)
    ax.set_title('Other renewables energy vs. DHI (8 measure locs median)')
    ax.set_ylabel('Other renewables energy [$MW$]')
    ax.set_xlabel('DHI [$W/m^2$]')

    fig, ax = plt.subplots()
    ax.scatter(joined['DNI'], joined['Other Renewables'], s=1)
    ax.set_title('Other renewables energy vs. DNI (8 measure locs median)')
    ax.set_ylabel('Other renewables energy [$MW$]')
    ax.set_xlabel('DNI [$W/m^2$]')

    lon = [-74.37, -72.57, -73.91, -78.45, -75.33, -73.97, -76.31, -73.61]
    lat = [42.3, 40.86, 41.14, 42.44, 43.10, 43.14, 43.38, 44.86]

    fig = plt.figure(figsize=(8, 8))
    m = Basemap(projection='cyl', resolution='i', area_thresh=100.,
                llcrnrlat=min(lat)-0.5, urcrnrlat=max(lat)+0.5,
                  llcrnrlon=min(lon)-2, urcrnrlon=max(lon)+0.8)
    m.fillcontinents(color='gray', lake_color='white')
    m.drawcoastlines(color='black', linewidth=1)
    m.drawcountries(color='black', linewidth=1)
    m.drawstates(color='black', linewidth=1)
    m.scatter(lon, lat, latlon=True, c='blue')
    plt.title('Location of the weather recordings')

    ##################################################
    # PREPROCESSED FUEL MIX DATA
    ##################################################
    fuel_mix = pd.read_csv('citylearn/data/nydata/fuelmix.csv')

    percent = np.array(fuel_mix['Renewable Share']) * 100
    fig, ax = plt.subplots()
    ax.plot(np.arange(percent.size), percent)
    ax.set_title('$\%$ Renweable Energy from total produced Energy')
    ax.set_ylabel('%')
    ax.set_xlabel('Time step')

    print(f'Min Renewable %: {percent.min()}')
    print(f'Max Renewable %: {percent.max()}')

    if save:
        filename = "../datasets/data_exploration_plots/ny-data-plots_" + datetime.now().strftime(
            "%Y%m%dT%H%M%S")
        save_multi_image(filename)
    else:
        plt.show()

##################################################
# ORIGINAL WEATHER DATA
##################################################
weather_orig = pd.read_csv('citylearn/data/citylearn_challenge_2022_phase_all/weather.csv')
dhi = np.array(weather_orig['Diffuse Solar Radiation [W/m2]'])
dni = np.array(weather_orig['Direct Solar Radiation [W/m2]'])

# Correlation DHI - Solar generation B1
solar_generation = np.array(
    pd.read_csv('citylearn/data/citylearn_challenge_2022_phase_all/Building_1.csv')['Solar Generation [W/kW]'])
fig, ax = plt.subplots()
ax.scatter(dhi, solar_generation, s=1)
ax.set_title('Solar generation Building 1 vs. DHI (original data)')
ax.set_ylabel('Solar generation [$W/kW$]')
ax.set_xlabel('DHI [$W/m^2$]')

fig, ax = plt.subplots()
ax.scatter(dni, solar_generation, s=1)
ax.set_title('Solar generation Building 1 vs. DNI (original data)')
ax.set_ylabel('Solar generation [$W/kW$]')
ax.set_xlabel('DNI [$W/m^2$]')

# Correlation DHI - Solar generation B6
solar_generation = np.array(
    pd.read_csv('citylearn/data/citylearn_challenge_2022_phase_all/Building_6.csv')['Solar Generation [W/kW]'])
fig, ax = plt.subplots()
ax.scatter(dhi, solar_generation, s=1)
ax.set_title('Solar generation Building 6 vs. DHI (original data)')
ax.set_ylabel('Solar generation [$W/kW$]')
ax.set_xlabel('DHI [$W/m^2$]')

fig, ax = plt.subplots()
ax.scatter(dni, solar_generation, s=1)
ax.set_title('Solar generation Building 6 vs. DNI (original data)')
ax.set_ylabel('Solar generation [$W/kW$]')
ax.set_xlabel('DNI [$W/m^2$]')

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

##################################################
# PREPROCESSED NY WEATHER DATA OWN BUILDINGS
##################################################
ny_weather_prep = pd.read_csv('citylearn/data/nydata/weather.csv')
dhi = np.array(ny_weather_prep['Diffuse Solar Radiation [W/m2]'])
dni = np.array(ny_weather_prep['Direct Solar Radiation [W/m2]'])

# Correlation DHI - Solar generation B1
solar_generation = np.array(
    pd.read_csv('citylearn/data/nydata_new_buildings/Building_1.csv')['Solar Generation [W/kW]'])
fig, ax = plt.subplots()
ax.scatter(dhi, solar_generation, s=1)
ax.set_title('Solar generation Building 1 vs. DHI (NY data, own buildings)')
ax.set_ylabel('Solar generation [$W/kW$]')
ax.set_xlabel('DHI [$W/m^2$]')

fig, ax = plt.subplots()
ax.scatter(dni, solar_generation, s=1)
ax.set_title('Solar generation Building 1 vs. DNI (NY data, own buildings)')
ax.set_ylabel('Solar generation [$W/kW$]')
ax.set_xlabel('DNI [$W/m^2$]')

# Correlation DHI - Solar generation B6
solar_generation = np.array(
    pd.read_csv('citylearn/data/nydata_new_buildings/Building_6.csv')['Solar Generation [W/kW]'])
fig, ax = plt.subplots()
ax.scatter(dhi, solar_generation, s=1)
ax.set_title('Solar generation Building 6 vs. DHI (NY data, own buildings)')
ax.set_ylabel('Solar generation [$W/kW$]')
ax.set_xlabel('DHI [$W/m^2$]')

fig, ax = plt.subplots()
ax.scatter(dni, solar_generation, s=1)
ax.set_title('Solar generation Building 6 vs. DNI (NY data, own buildings)')
ax.set_ylabel('Solar generation [$W/kW$]')
ax.set_xlabel('DNI [$W/m^2$]')

# Sum produced solar energy new buildings vs old buildings
buildings_filenames_old = glob.glob('citylearn/data/nydata/Building_*.csv')
buildings_filenames_new = glob.glob('citylearn/data/nydata_new_buildings/Building_*.csv')

all_buildings_old = []
all_buildings_new = []
for i, file in enumerate(buildings_filenames_old):
    all_buildings_old.append(pd.read_csv(file))
    all_buildings_new.append(pd.read_csv(buildings_filenames_new[i]))

all_buildings_old_pvprod = pd.concat(all_buildings_old).reset_index().groupby('index', as_index=False).sum()[
    'Solar Generation [W/kW]']
all_buildings_new_pvprod = pd.concat(all_buildings_new).reset_index().groupby('index', as_index=False).sum()[
    'Solar Generation [W/kW]']

fig, ax = plt.subplots()
ax.plot(np.arange(len(all_buildings_old_pvprod)), all_buildings_old_pvprod, label='Original buildings')
ax.plot(np.arange(len(all_buildings_new_pvprod)), all_buildings_new_pvprod, label='Own buildings')
ax.axhline(all_buildings_old_pvprod.mean(), c='red')
ax.text(0, all_buildings_old_pvprod.mean()+0.1, 'Mean of original buildings', rotation=0)
ax.axhline(all_buildings_new_pvprod.mean(), c='red')
ax.text(0, all_buildings_new_pvprod.mean()+0.1, 'Mean of own buildings', rotation=0)
ax.set_title('Sum of PV production of all buildings')
ax.set_xlabel('Time step')
ax.set_ylabel('PV production [W/kW]')
ax.legend()

if save:
    filename = "../datasets/data_exploration_plots/buildings-pv-production-plots_" + datetime.now().strftime(
        "%Y%m%dT%H%M%S")
    save_multi_image(filename)
else:
    plt.show()

##################################################
# PRICING DATA
##################################################

pricing_new = pd.read_csv('citylearn/data/nydata/pricing.csv')['Electricity Pricing [$]']

fossil_share = 1 - fuel_mix['Renewable Share']
alpha = 20
pricing_new = pricing_new + alpha * fossil_share

# normalize between 0 and 1
pricing_new = (pricing_new - np.min(pricing_new)) / (np.max(pricing_new) - np.min(pricing_new))

fig, ax = plt.subplots()
ax.plot(np.arange(len(pricing_new)), pricing_new)
ax.set_title(f'Electricity Pricing [$] - Weighted by Fossil Energy share (factor = {alpha})')
ax.set_ylabel('$')
ax.set_xlabel('Time step')

fig, ax = plt.subplots()
ax.plot(np.arange(168), pricing_new[:168], label='Prices influenced by fossil energy')
ax.plot(np.arange(168), pricing[:168], label='Old prices')
ax.set_title(f'Electricity Pricing [$] (factor = {alpha})')
ax.set_ylabel('$')
ax.set_xticks(np.arange(0, len(ticks)))
ax.xaxis.set_tick_params(length=0)
ax.set_xticklabels(ticks, rotation=0)
ax.legend()
[l.set_visible(False) for (i, l) in enumerate(ax.get_xticklabels()) if (i - 18) % 24 != 0]

fig, ax = plt.subplots()
ax.scatter(fuel_mix['Renewable Share'], pricing_new, s=1)
ax.set_title('Weighted electricity price vs. renewable share')
ax.set_xlabel('Renewable Share')
ax.set_ylabel('Weighted electricity price [$]')

solar_generation = np.array(pd.read_csv('citylearn/data/nydata/Building_1.csv')['Solar Generation [W/kW]'])
fig, ax = plt.subplots()
ax.scatter(solar_generation, pricing_new, s=1)
ax.set_title('Weighted electricity price vs. solar generation B1 (NY data)')
ax.set_xlabel('Solar Generation of Building 1 [$W/kW$]')
ax.set_ylabel('Weighted electricity price [$]')

solar_generation = np.array(
    pd.read_csv('citylearn/data/nydata_new_buildings/Building_1.csv')['Solar Generation [W/kW]'])
fig, ax = plt.subplots()
ax.scatter(solar_generation, pricing_new, s=1)
ax.set_title('Weighted electricity price vs. solar generation B1 (NY data, own buildings)')
ax.set_xlabel('Solar Generation of Building 1 [$W/kW$]')
ax.set_ylabel('Weighted electricity price [$]')

if save:
    filename = "../datasets/data_exploration_plots/pricing-plots_" + datetime.now().strftime("%Y%m%dT%H%M%S")
    save_multi_image(filename)
else:
    plt.show()
