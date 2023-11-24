import glob
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

from mpl_toolkits.basemap import Basemap
from scipy.stats import pearsonr

from utils import save_multi_image

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

def create_scatter_plot(ax, x, y):
    ax.scatter(x, y, s=1)
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m * x + b, color='green')


def get_ticks():
    ticks = pd.Series(pd.date_range(start=f'2021-01-01', end=f'2021-01-01 23:00:00', freq='H')).dt.time.astype(
        str).tolist() * 7
    ticks = ['Mo'] * 24
    ticks.extend(['Tue'] * 24)
    ticks.extend(['Wed'] * 24)
    ticks.extend(['Thur'] * 24)
    ticks.extend(['Fri'] * 24)
    ticks.extend(['Sat'] * 24)
    ticks.extend(['Sun'] * 24)

    return ticks


##################################################
# DATA CITYLEARN CHALLENGE 2022
##################################################

def analyze_challenge_data(save, timestamp):
    weather = pd.read_csv('../citylearn/data/citylearn_challenge_2022_phase_all/weather.csv')

    # normalize carbon intensity between 0 and 1
    carbon_intensity = pd.read_csv('../citylearn/data/citylearn_challenge_2022_phase_all/carbon_intensity.csv')[
        'kg_CO2/kWh']
    carbon_intensity = (carbon_intensity - np.min(carbon_intensity)) / (
            np.max(carbon_intensity) - np.min(carbon_intensity))

    # normalize prices between 0 and 1
    pricing = pd.read_csv('../citylearn/data/citylearn_challenge_2022_phase_all/pricing.csv')['Electricity Pricing [$]']
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
    create_scatter_plot(ax2, direct_solar, carbon_intensity)
    ax2.set_title('Direct solar radiation vs. carbon intensity')
    ax2.set_ylabel('Carbon intensity [$kg_CO2/kWh$]')
    ax2.set_xlabel('Direct solar radiation [$W/M^2$]')

    fig5, ax5 = plt.subplots()
    create_scatter_plot(ax5, direct_solar, diffuse_solar)
    ax5.set_title('Diffuse solar radiation vs. direct solar radiation')
    ax5.set_xlabel('Diffuse solar radiation [$W/M^2$]')
    ax5.set_ylabel('Direct solar radiation [$W/M^2$]')

    fig1, ax1 = plt.subplots()
    ax1.plot(xs[:168], pricing[0:168])
    ax1.set_title('Electricity Pricing of one week')
    ax1.set_ylabel('Electricity Price [$]')
    ticks = get_ticks()
    ax1.set_xticks(np.arange(0, len(ticks)))
    ax1.xaxis.set_tick_params(length=0)
    ax1.set_xticklabels(ticks, rotation=0)
    [l.set_visible(False) for (i, l) in enumerate(ax1.get_xticklabels()) if (i - 18) % 24 != 0]

    if save:
        filename = "../datasets/data_exploration_plots/citylearn-challenge-2022-plots_" + timestamp
        save_multi_image(filename)
    else:
        plt.show()


##################################################
# REAL DATA FROM NEW YORK ISO AND NREL
##################################################

def analyze_ny_data(save, timestamp):
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
    create_scatter_plot(ax, joined['Wind Speed'], joined['Wind'])
    ax.set_title('Wind energy vs. wind speed')
    ax.set_ylabel('Wind energy [$MW$]')
    ax.set_xlabel('Wind speed [$m/s$]')

    fig, ax = plt.subplots()
    create_scatter_plot(ax, joined['DHI'], joined['Other Renewables'])
    ax.set_title('Other renewables energy vs. DHI')
    ax.set_ylabel('Other renewables energy [$MW$]')
    ax.set_xlabel('DHI [$W/m^2$]')

    fig, ax = plt.subplots()
    create_scatter_plot(ax, joined['DNI'], joined['Other Renewables'])
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
    create_scatter_plot(ax, joined['Wind Speed'], joined['Wind'])
    ax.set_title('Wind energy vs. wind speed (8 measure locs median)')
    ax.set_ylabel('Wind energy [$MW$]')
    ax.set_xlabel('Wind speed [$m/s$]')

    fig, ax = plt.subplots()
    create_scatter_plot(ax, joined['DHI'], joined['Other Renewables'])
    ax.set_title('Other renewables energy vs. DHI (8 measure locs median)')
    ax.set_ylabel('Other renewables energy [$MW$]')
    ax.set_xlabel('DHI [$W/m^2$]')

    fig, ax = plt.subplots()
    create_scatter_plot(ax, joined['DNI'], joined['Other Renewables'])
    ax.set_title('Other renewables energy vs. DNI (8 measure locs median)')
    ax.set_ylabel('Other renewables energy [$MW$]')
    ax.set_xlabel('DNI [$W/m^2$]')

    lon = [-74.37, -72.57, -73.91, -78.45, -75.33, -73.97, -76.31, -73.61]
    lat = [42.3, 40.86, 41.14, 42.44, 43.10, 43.14, 43.38, 44.86]

    fig = plt.figure(figsize=(8, 8))
    m = Basemap(projection='cyl', resolution='i', area_thresh=100.,
                llcrnrlat=min(lat) - 0.5, urcrnrlat=max(lat) + 0.5,
                llcrnrlon=min(lon) - 2, urcrnrlon=max(lon) + 0.8)
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
    print(f'Mean Renewable %: {percent.mean()}')
    print(f'Median Renewable %: {np.median(percent)}')

    if save:
        filename = "../datasets/data_exploration_plots/ny-data-plots_" + timestamp
        save_multi_image(filename)
    else:
        plt.show()


##################################################
# ORIGINAL WEATHER DATA
##################################################
def analyze_challenge_weather_data():
    weather_orig = pd.read_csv('../citylearn/data/citylearn_challenge_2022_phase_all/weather.csv')
    dhi = np.array(weather_orig['Diffuse Solar Radiation [W/m2]'])
    dni = np.array(weather_orig['Direct Solar Radiation [W/m2]'])

    # Correlation DHI - Solar generation B1
    solar_generation = np.array(
        pd.read_csv('../citylearn/data/citylearn_challenge_2022_phase_all/Building_1.csv')['Solar Generation [W/kW]'])
    fig, ax = plt.subplots()
    create_scatter_plot(ax, dhi, solar_generation)
    ax.set_title('Solar generation Building 1 vs. DHI (original data)')
    ax.set_ylabel('Solar generation [$W/kW$]')
    ax.set_xlabel('DHI [$W/m^2$]')

    fig, ax = plt.subplots()
    create_scatter_plot(ax, dni, solar_generation)
    ax.set_title('Solar generation Building 1 vs. DNI (original data)')
    ax.set_ylabel('Solar generation [$W/kW$]')
    ax.set_xlabel('DNI [$W/m^2$]')

    # Correlation DHI - Solar generation B6
    solar_generation = np.array(
        pd.read_csv('../citylearn/data/citylearn_challenge_2022_phase_all/Building_6.csv')['Solar Generation [W/kW]'])
    fig, ax = plt.subplots()
    create_scatter_plot(ax, dhi, solar_generation)
    ax.set_title('Solar generation Building 6 vs. DHI (original data)')
    ax.set_ylabel('Solar generation [$W/kW$]')
    ax.set_xlabel('DHI [$W/m^2$]')

    fig, ax = plt.subplots()
    create_scatter_plot(ax, dni, solar_generation)
    ax.set_title('Solar generation Building 6 vs. DNI (original data)')
    ax.set_ylabel('Solar generation [$W/kW$]')
    ax.set_xlabel('DNI [$W/m^2$]')


##################################################
# PREPROCESSED NY WEATHER DATA
##################################################
def analyze_building_weather_correlation():
    ny_weather_prep = pd.read_csv('../citylearn/data/nydata/weather.csv')
    dhi = np.array(ny_weather_prep['Diffuse Solar Radiation [W/m2]'])
    dni = np.array(ny_weather_prep['Direct Solar Radiation [W/m2]'])

    # Correlation DHI - Solar generation B1
    solar_generation = np.array(pd.read_csv('../citylearn/data/nydata/Building_1.csv')['Solar Generation [W/kW]'])
    fig, ax = plt.subplots()
    create_scatter_plot(ax, dhi, solar_generation)
    ax.set_title('Solar generation Building 1 vs. DHI (NY data)')
    ax.set_ylabel('Solar generation [$W/kW$]')
    ax.set_xlabel('DHI [$W/m^2$]')

    fig, ax = plt.subplots()
    create_scatter_plot(ax, dni, solar_generation)
    ax.set_title('Solar generation Building 1 vs. DNI (NY data)')
    ax.set_ylabel('Solar generation [$W/kW$]')
    ax.set_xlabel('DNI [$W/m^2$]')

    # Correlation DHI - Solar generation B6
    solar_generation = np.array(pd.read_csv('../citylearn/data/nydata/Building_6.csv')['Solar Generation [W/kW]'])
    fig, ax = plt.subplots()
    create_scatter_plot(ax, dhi, solar_generation)
    ax.set_title('Solar generation Building 6 vs. DHI (NY data)')
    ax.set_ylabel('Solar generation [$W/kW$]')
    ax.set_xlabel('DHI [$W/m^2$]')

    fig, ax = plt.subplots()
    create_scatter_plot(ax, dni, solar_generation)
    ax.set_title('Solar generation Building 6 vs. DNI (NY data)')
    ax.set_ylabel('Solar generation [$W/kW$]')
    ax.set_xlabel('DNI [$W/m^2$]')


##################################################
# PREPROCESSED NY WEATHER DATA OWN BUILDINGS
##################################################
def analyze_building_weather_correlation_own():
    ny_weather_prep = pd.read_csv('citylearn/data/nydata/weather.csv')
    dhi = np.array(ny_weather_prep['Diffuse Solar Radiation [W/m2]'])
    dni = np.array(ny_weather_prep['Direct Solar Radiation [W/m2]'])

    renewable = pd.read_csv('citylearn/data/nnb_limitobs1/fuelmix.csv')['Renewable Sources [kWh]']

    # Correlation DHI - Solar generation B1
    solar_generation = np.array(
        pd.read_csv('citylearn/data/nydata_new_buildings2/Building_1.csv')['Solar Generation [W/kW]'])
    fig, ax = plt.subplots()
    create_scatter_plot(ax, dhi, solar_generation)
    ax.set_title('Solar generation Building 1 vs. DHI (NY data, own buildings2)')
    ax.set_ylabel('Solar generation [$W/kW$]')
    ax.set_xlabel('DHI [$W/m^2$]')

    fig, ax = plt.subplots()
    create_scatter_plot(ax, dni, solar_generation)
    ax.set_title('Solar generation Building 1 vs. DNI (NY data, own buildings2)')
    ax.set_ylabel('Solar generation [$W/kW$]')
    ax.set_xlabel('DNI [$W/m^2$]')

    fig, ax = plt.subplots()
    create_scatter_plot(ax, renewable, solar_generation)
    ax.set_title('Solar generation Building 1 vs. Renewable production (NY data)')
    ax.set_ylabel('Solar generation [$W/kW$]')
    ax.set_xlabel('Renewable production [$kW$]')

    # Correlation DHI - Solar generation B6
    solar_generation = np.array(
        pd.read_csv('citylearn/data/nydata_new_buildings2/Building_6.csv')['Solar Generation [W/kW]'])
    fig, ax = plt.subplots()
    create_scatter_plot(ax, dhi, solar_generation)
    ax.set_title('Solar generation Building 6 vs. DHI (NY data, own buildings2)')
    ax.set_ylabel('Solar generation [$W/kW$]')
    ax.set_xlabel('DHI [$W/m^2$]')

    fig, ax = plt.subplots()
    create_scatter_plot(ax, dni, solar_generation)
    ax.set_title('Solar generation Building 6 vs. DNI (NY data, own buildings2)')
    ax.set_ylabel('Solar generation [$W/kW$]')
    ax.set_xlabel('DNI [$W/m^2$]')

    fig, ax = plt.subplots()
    create_scatter_plot(ax, renewable, solar_generation)
    ax.set_title('Solar generation Building 6 vs. Renewable production (NY data)')
    ax.set_ylabel('Solar generation [$W/kW$]')
    ax.set_xlabel('Renewable production [$kW$]')


##################################################
# PREPROCESSED NY WEATHER DATA 8 LOCS OWN BUILDINGS
##################################################
def analyze_building_8locsweather_correlation_own(save, timestamp):
    ny_weather_prep = pd.read_csv('citylearn/data/nydata/weather_8locs_median.csv')
    dhi = np.array(ny_weather_prep['Diffuse Solar Radiation [W/m2]'])
    dni = np.array(ny_weather_prep['Direct Solar Radiation [W/m2]'])

    # Correlation DHI - Solar generation B1
    solar_generation = np.array(
        pd.read_csv('citylearn/data/nydata_new_buildings2/Building_1.csv')['Solar Generation [W/kW]'])
    fig, ax = plt.subplots()
    create_scatter_plot(ax, dhi, solar_generation)
    ax.set_title('Solar generation Building 1 vs. DHI (NY data 8 locs median, own buildings2)')
    ax.set_ylabel('Solar generation [$W/kW$]')
    ax.set_xlabel('DHI [$W/m^2$]')

    fig, ax = plt.subplots()
    create_scatter_plot(ax, dni, solar_generation)
    ax.set_title('Solar generation Building 1 vs. DNI (NY data 8 locs median, own buildings2)')
    ax.set_ylabel('Solar generation [$W/kW$]')
    ax.set_xlabel('DNI [$W/m^2$]')

    # Correlation DHI - Solar generation B6
    solar_generation = np.array(
        pd.read_csv('citylearn/data/nydata_new_buildings2/Building_6.csv')['Solar Generation [W/kW]'])
    fig, ax = plt.subplots()
    create_scatter_plot(ax, dhi, solar_generation)
    ax.set_title('Solar generation Building 6 vs. DHI (NY data 8 locs median, own buildings2)')
    ax.set_ylabel('Solar generation [$W/kW$]')
    ax.set_xlabel('DHI [$W/m^2$]')

    fig, ax = plt.subplots()
    create_scatter_plot(ax, dni, solar_generation)
    ax.set_title('Solar generation Building 6 vs. DNI (NY data 8 locs median, own buildings2)')
    ax.set_ylabel('Solar generation [$W/kW$]')
    ax.set_xlabel('DNI [$W/m^2$]')

    # Sum produced solar energy new buildings vs old buildings
    buildings_filenames_old = glob.glob('citylearn/data/nydata/Building_*.csv')
    buildings_filenames_new = glob.glob('citylearn/data/nydata_new_buildings/Building_*.csv')
    buildings_filenames_new2 = glob.glob('citylearn/data/nydata_new_buildings2/Building_*.csv')

    all_buildings_old = []
    all_buildings_new = []
    all_buildings_new2 = []
    for i, file in enumerate(buildings_filenames_old):
        all_buildings_old.append(pd.read_csv(file))
        all_buildings_new.append(pd.read_csv(buildings_filenames_new[i]))
        all_buildings_new2.append(pd.read_csv(buildings_filenames_new2[i]))

    all_buildings_old_pvprod = pd.concat(all_buildings_old).reset_index().groupby('index', as_index=False).sum()[
        'Solar Generation [W/kW]']
    all_buildings_new_pvprod = pd.concat(all_buildings_new).reset_index().groupby('index', as_index=False).sum()[
        'Solar Generation [W/kW]']
    all_buildings_new2_pvprod = pd.concat(all_buildings_new2).reset_index().groupby('index', as_index=False).sum()[
        'Solar Generation [W/kW]']

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(all_buildings_old_pvprod)), all_buildings_old_pvprod, label='Original buildings')
    ax.plot(np.arange(len(all_buildings_new_pvprod)), all_buildings_new_pvprod, label='Own buildings')
    ax.plot(np.arange(len(all_buildings_new2_pvprod)), all_buildings_new2_pvprod, label='Own buildings 2')
    ax.axhline(all_buildings_old_pvprod.mean(), c='red')
    ax.text(0, all_buildings_old_pvprod.mean() + 0.1, f'Mean of original buildings {all_buildings_old_pvprod.mean()}',
            rotation=0)
    ax.axhline(all_buildings_new_pvprod.mean(), c='red')
    ax.text(0, all_buildings_new_pvprod.mean() + 0.1, f'Mean of own buildings {all_buildings_new_pvprod.mean()}',
            rotation=0)
    ax.axhline(all_buildings_new2_pvprod.mean(), c='red')
    ax.text(0, all_buildings_new2_pvprod.mean() + 0.1, f'Mean of own buildings 2 {all_buildings_new2_pvprod.mean()}',
            rotation=0)
    ax.set_title('Sum of PV production of all buildings')
    ax.set_xlabel('Time step')
    ax.set_ylabel('PV production [W/kW]')
    ax.legend()

    if save:
        filename = "../datasets/data_exploration_plots/buildings-pv-production-plots_" + timestamp
        save_multi_image(filename)
    else:
        plt.show()


##################################################
# PRICING DATA
##################################################
def analyze_pricing_data(save, timestamp):
    ticks = get_ticks()

    pricing_new = pd.read_csv('../citylearn/data/nydata/pricing.csv')['Electricity Pricing [$]']
    fuel_mix = pd.read_csv('../citylearn/data/nydata/fuelmix.csv')

    fossil_share = 1 - fuel_mix['Renewable Share']
    alpha = 20
    pricing_new = pricing_new + alpha * fossil_share

    # normalize between 0 and 1
    pricing_new = (pricing_new - np.min(pricing_new)) / (np.max(pricing_new) - np.min(pricing_new))
    pricing = pd.read_csv('../citylearn/data/citylearn_challenge_2022_phase_all/pricing.csv')['Electricity Pricing [$]']

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
    create_scatter_plot(ax, fuel_mix['Renewable Share'], pricing_new)
    ax.set_title('Weighted electricity price vs. renewable share')
    ax.set_xlabel('Renewable Share')
    ax.set_ylabel('Weighted electricity price [$]')

    solar_generation = np.array(pd.read_csv('../citylearn/data/nydata/Building_1.csv')['Solar Generation [W/kW]'])
    fig, ax = plt.subplots()
    create_scatter_plot(ax, solar_generation, pricing_new)
    ax.set_title('Weighted electricity price vs. solar generation B1 (NY data)')
    ax.set_xlabel('Solar Generation of Building 1 [$W/kW$]')
    ax.set_ylabel('Weighted electricity price [$]')

    solar_generation = np.array(
        pd.read_csv('../citylearn/data/nydata_new_buildings2/Building_1.csv')['Solar Generation [W/kW]'])
    fig, ax = plt.subplots()
    create_scatter_plot(ax, solar_generation, pricing_new)
    ax.set_title('Weighted electricity price vs. solar generation B1 (NY data, own buildings2)')
    ax.set_xlabel('Solar Generation of Building 1 [$W/kW$]')
    ax.set_ylabel('Weighted electricity price [$]')

    if save:
        filename = "../datasets/data_exploration_plots/pricing-plots_" + timestamp
        save_multi_image(filename)
    else:
        plt.show()


def analyze_own_building_data(save, timestamp):
    y_min_gen = 10000000000000000
    y_max_gen = -10000000

    nominal_power = [4.0, 4.0, 4.0, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

    for i in range(1, 18):
        data = pd.read_csv(f'citylearn/data/nydata_new_buildings2/Building_{i}.csv')

        fig, ax = plt.subplots(nrows=2, sharex=True)
        fig.set_figheight(8)
        fig.set_figwidth(13)
        fig.suptitle(f'Building {i}')
        ax[0].plot(data['Equipment Electric Power [kWh]'])
        ax[0].set_ylabel(f'Non-shiftable Load [kWh]')
        ax[0].set_ylim([0, 9])

        ax[1].plot(data['Solar Generation [W/kW]'])
        ax[1].set_ylabel(f'Solar Generation [W/kW]')
        ax[1].set_xlabel(f'Time step')
        ax[1].set_ylim([0, 1400])

        fig.tight_layout()

        fig, ax = plt.subplots()
        fig.set_figheight(5)
        fig.set_figwidth(13)
        fig.suptitle(f'Building {i} - Load vs. Solar Generation')
        ax.plot(data['Equipment Electric Power [kWh]'], label='load')
        ax.plot(nominal_power[i - 1] * np.array(data['Solar Generation [W/kW]']) / 1000, label='generation')
        ax.legend()
        ax.set_ylabel(f'kWh')
        ax.set_xlabel(f'Time step')
        ax.set_ylim([0, 9])

        if (nominal_power[i - 1] * np.array(data['Solar Generation [W/kW]']) / 1000).min() < y_min_gen:
            y_min_gen = (nominal_power[i - 1] * np.array(data['Solar Generation [W/kW]']) / 1000).min()
        if (nominal_power[i - 1] * np.array(data['Solar Generation [W/kW]']) / 1000).max() > y_max_gen:
            y_max_gen = (nominal_power[i - 1] * np.array(data['Solar Generation [W/kW]']) / 1000).max()

    if save:
        filename = "../datasets/data_exploration_plots/building-plots_" + timestamp
        save_multi_image(filename)
    else:
        plt.show()


def plot_building_means(save, timestamp):
    y_min_gen = 10000000000000000
    y_max_gen = -10000000

    nominal_power = [4.0, 4.0, 4.0, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

    fig, axs = plt.subplots(17, figsize=(13, 17), sharex=True)
    # fig.suptitle(f'Daily Mean Load and Solar Generation of all Buildings')

    for i in range(1, 18):
        data = pd.read_csv(f'citylearn/data/nnb_limitobs1/Building_{i}.csv')
        data = data.groupby(np.arange(len(data)) // 24).mean()

        if (nominal_power[i - 1] * np.array(data['Solar Generation [W/kW]']) / 1000).min() < y_min_gen:
            y_min_gen = (nominal_power[i - 1] * np.array(data['Solar Generation [W/kW]']) / 1000).min()
        if (nominal_power[i - 1] * np.array(data['Solar Generation [W/kW]']) / 1000).max() > y_max_gen:
            y_max_gen = (nominal_power[i - 1] * np.array(data['Solar Generation [W/kW]']) / 1000).max()

        axs[i - 1].plot(data['Equipment Electric Power [kWh]'], label='load')
        axs[i - 1].plot(nominal_power[i - 1] * np.array(data['Solar Generation [W/kW]']) / 1000,
                        label='solar generation')
        # axs[i-1].legend()
        axs[i - 1].set_ylabel(f'B{i}', fontsize=19, rotation=0, labelpad=43, loc='bottom')
        # axs[i-1].set_xlabel(f'Time step')
        axs[i - 1].set_ylim([0, 4.3])
        axs[i - 1].set_xlim([-1, 365])
        axs[i - 1].grid()

        if not i == 17:
            axs[i - 1].set_xticks([])
        axs[i - 1].tick_params(axis='x', which='both', labelsize=19)
        axs[i - 1].tick_params(axis='y', which='both', labelsize=15)
        # axs[i - 1].set_yticks([])

    handles, labels = axs[0].get_legend_handles_labels()
    # plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xticks(np.linspace(0, 365, 13)[:-1],
               ('Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'))
    fig.legend(handles, labels, loc='upper center', fontsize=21, framealpha=0)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.subplots_adjust(wspace=0, hspace=0.1)

    if save:
        filename = "../datasets/data_exploration_plots/building-plots-mean_" + timestamp
        save_multi_image(filename)
    else:
        plt.show()


def get_fuel_data():
    load_dir = '../datasets/fuel_mix_ny_2021'
    fuel_files = os.listdir(load_dir)
    fuel = pd.read_csv(f'{load_dir}/{fuel_files[0]}')
    fuel = (fuel.pivot_table(index=['Time Stamp'],
                             columns='Fuel Category',
                             values='Gen MW').reset_index().rename_axis(None, axis=1))

    for file in fuel_files[1:]:
        new = pd.read_csv(f'{load_dir}/{file}')
        new = (new.pivot_table(index=['Time Stamp'],
                               columns='Fuel Category',
                               values='Gen MW').reset_index().rename_axis(None, axis=1))
        fuel = fuel.append(new)

    fuel = fuel.sort_values('Time Stamp')

    fuel['Year'] = pd.DatetimeIndex(fuel['Time Stamp']).year
    fuel['Month'] = pd.DatetimeIndex(fuel['Time Stamp']).month
    fuel['Day'] = pd.DatetimeIndex(fuel['Time Stamp']).day
    fuel['Hour'] = pd.DatetimeIndex(fuel['Time Stamp']).hour
    fuel['Minute'] = pd.DatetimeIndex(fuel['Time Stamp']).minute

    # group by renewable and not renewable and convert given MW into kWh
    fuel['Renewable Sources [kW]'] = fuel[['Hydro', 'Wind', 'Other Renewables']].sum(axis=1) * 1000
    fuel['Other [kW]'] = fuel[['Dual Fuel', 'Natural Gas', 'Nuclear', 'Other Fossil Fuels']].sum(axis=1) * 1000

    df = fuel.groupby(by=['Year', 'Month', 'Day', 'Hour']).sum().reset_index()
    df['Renewable Sources [kWh]'] = 1 / 12 * df['Renewable Sources [kW]']
    df['Other [kWh]'] = 1 / 12 * df['Other [kW]']
    df['Hydro'] = 1 / 12 * (df['Hydro'] * 1000)
    df['Wind'] = 1 / 12 * (df['Wind'] * 1000)
    df['Other Renewables'] = 1 / 12 * (df['Other Renewables'] * 1000)
    df['Hour'] = df['Hour'] + 1

    df['Renewable Share'] = df['Renewable Sources [kWh]'] / (df['Renewable Sources [kWh]'] + df['Other [kWh]'])
    df = df.drop(columns=['Renewable Sources [kW]', 'Other [kW]', 'Other [kWh]'])

    df['Datetime'] = pd.to_datetime(df[['Month', 'Day', 'Year']].astype(str).apply(' '.join, 1),
                                    format='%m %d %Y')

    df['Day Type'] = df['Datetime'].dt.dayofweek
    df['Day Type'] = df['Day Type'] + 2
    df.loc[df['Day Type'] == 8, ['Day Type']] = 1

    first_row_idx = df.index[(df['Month'] == 7) &
                             (df['Hour'] == 24) &
                             (df['Day Type'] == 7)].tolist()[-1]
    idx = df.index.tolist()
    del idx[:first_row_idx]
    df = df.reindex(idx + list(range(first_row_idx)))

    return df


def weather_correlations(save, timestamp):
    ny_weather_prep = pd.read_csv('citylearn/data/nnb_limitobs1/weather_8locs_median.csv')
    fuelmix = get_fuel_data()

    print('Correlation Wind speed and Wind energy', pearsonr(ny_weather_prep['Wind Speed [m/s]'],
                                                             fuelmix['Wind'])[0])
    print('Correlation DHI and Other renewable', pearsonr(ny_weather_prep['Diffuse Solar Radiation [W/m2]'],
                                                          fuelmix['Other Renewables'])[0])
    print('Correlation DNI and Other renewable', pearsonr(ny_weather_prep['Direct Solar Radiation [W/m2]'],
                                                          fuelmix['Other Renewables'])[0])
    print('Correlation Wind speed and Total renewable', pearsonr(ny_weather_prep['Wind Speed [m/s]'],
                                                                 fuelmix['Renewable Sources [kWh]'])[0])
    print('Correlation DHI and Total renewable', pearsonr(ny_weather_prep['Diffuse Solar Radiation [W/m2]'],
                                                          fuelmix['Renewable Sources [kWh]'])[0])
    print('Correlation DNI and Total renewable', pearsonr(ny_weather_prep['Direct Solar Radiation [W/m2]'],
                                                          fuelmix['Renewable Sources [kWh]'])[0])


def plot_weather_means(save, timestamp):
    ny_weather_prep = pd.read_csv('citylearn/data/nnb_limitobs1/weather_8locs_median.csv')
    fuelmix = get_fuel_data()

    ny_weather_prep = ny_weather_prep.groupby(np.arange(len(ny_weather_prep)) // 24).mean()
    fuelmix = fuelmix.groupby(np.arange(len(fuelmix)) // 24).mean()

    print('Hydro generated', fuelmix['Hydro'].sum())
    print('Hydro share', fuelmix['Hydro'].sum() / fuelmix['Renewable Sources [kWh]'].sum())
    print('Wind generated', fuelmix['Wind'].sum())
    print('Wind share', fuelmix['Wind'].sum() / fuelmix['Renewable Sources [kWh]'].sum())
    print('Other generated', fuelmix['Other Renewables'].sum())
    print('Other share', fuelmix['Other Renewables'].sum() / fuelmix['Renewable Sources [kWh]'].sum())

    cols = {'Diffuse Solar Radiation [W/m2]': 'DHI',
            'Direct Solar Radiation [W/m2]': 'DNI',
            'Wind Speed [m/s]': 'Wind Speed',
            'Outdoor Drybulb Temperature [C]': 'Temperature',
            'Relative Humidity [%]': 'Relative Humidity',
            'Wind': 'Wind Generation',
            'Renewable Sources [kWh]': 'Total Renewable Generation',
            'Renewable Share': 'Renewable Share', }

    units = ['$W/m^2$', '$W/m^2$', '$m/s$', '$°C$', '$\%$', '$kWh$', '$kWh$', '$\%$']

    fig, axs = plt.subplots(len(cols), figsize=(14, 17), sharex=True)

    for i, name in enumerate(cols):
        if name == 'Renewable Share':
            axs[i].plot(fuelmix[name] * 100)
        elif name in ['Renewable Sources [kWh]', 'Wind']:
            axs[i].plot(fuelmix[name])
        else:
            axs[i].plot(ny_weather_prep[name], label=name)
        axs[i].set_title(cols[name], fontsize=19, rotation=0,  # labelpad=60,
                         loc='center')
        axs[i].set_ylabel(units[i], fontsize=19)
        # axs[i-1].set_xlabel(f'Time step')
        # axs[i].set_ylim([0, 4.3])
        axs[i].set_xlim([-1, 365])
        axs[i - 1].grid()

        if not i == len(cols) + 1:
            axs[i - 1].set_xticks([])
        axs[i - 1].tick_params(axis='x', which='both', labelsize=19)
        axs[i - 1].tick_params(axis='y', which='both', labelsize=17)
        # axs[i - 1].set_yticks([])

    plt.xticks(np.linspace(0, 365, 13)[:-1],
               ('Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul'))
    plt.tight_layout(rect=[0, 0, 1, 1])
    fig.subplots_adjust(wspace=0, hspace=0.3)

    if save:
        filename = "../datasets/data_exploration_plots/weather-mean_" + timestamp
        save_multi_image(filename)
    else:
        plt.show()


if __name__ == '__main__':
    save = True
    challenge_data = False
    ny_data = False
    pricing_data = False
    building_data = False
    building_data_means = True
    weather_means = False
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    if challenge_data:
        analyze_challenge_data(save, timestamp)
        analyze_challenge_weather_data(save, timestamp)
        analyze_building_weather_correlation()
    if ny_data:
        analyze_ny_data(save, timestamp)
        analyze_building_weather_correlation_own()
        analyze_building_8locsweather_correlation_own(save, timestamp)
    if pricing_data:
        analyze_pricing_data(save, timestamp)
    if building_data:
        analyze_own_building_data(save, timestamp)
    if building_data_means:
        plot_building_means(save, timestamp)
    if weather_means:
        plot_weather_means(save, timestamp)
