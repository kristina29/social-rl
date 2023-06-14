import os

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

WEATHER_VARS = ['Outdoor Drybulb Temperature [C]',
                'Relative Humidity [%]',
                'Diffuse Solar Radiation [W/m2]',
                'Direct Solar Radiation [W/m2]',
                'Wind Speed [m/s]']

WEATHER_FINAL_ORDER = ['Wind Speed [m/s]',
                       '6h Prediction Wind Speed [m/s]',
                       '12h Prediction Wind Speed [m/s]',
                       '24h Prediction Wind Speed [m/s]',
                       'Outdoor Drybulb Temperature [C]',
                       'Relative Humidity [%]',
                       'Diffuse Solar Radiation [W/m2]',
                       'Direct Solar Radiation [W/m2]',
                       '6h Prediction Outdoor Drybulb Temperature [C]',
                       '12h Prediction Outdoor Drybulb Temperature [C]',
                       '24h Prediction Outdoor Drybulb Temperature [C]',
                       '6h Prediction Relative Humidity [%]',
                       '12h Prediction Relative Humidity [%]',
                       '24h Prediction Relative Humidity [%]',
                       '6h Prediction Diffuse Solar Radiation [W/m2]',
                       '12h Prediction Diffuse Solar Radiation [W/m2]',
                       '24h Prediction Diffuse Solar Radiation [W/m2]',
                       '6h Prediction Direct Solar Radiation [W/m2]',
                       '12h Prediction Direct Solar Radiation [W/m2]',
                       '24h Prediction Direct Solar Radiation [W/m2]']


def preprocess_data(load_path, save_path, weather) -> pd.DataFrame:
    if weather:
        df = pd.read_csv(load_path, skiprows=2)

        # rename columns to match citylearn framework
        df = df.rename(columns={'Temperature': WEATHER_VARS[0],
                                'Relative Humidity': WEATHER_VARS[1],
                                'DHI': WEATHER_VARS[2],
                                'DNI': WEATHER_VARS[3],
                                'Wind Speed': WEATHER_VARS[4]})
    else:
        df = read_fuel_data(load_path)


    df = median_by_hour(df)
    df = add_daytype(df)
    df = correct_time_series_start(df)

    if weather:
        df = add_predictions(df)

    # drop not needed columns
    df = df.drop(columns=['Datetime', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Day Type'])

    if weather:
        # reorder to match citylearn
        df = df[WEATHER_FINAL_ORDER]

    df.to_csv(save_path, index=False)
    print(f'File saved under {save_path}')

    return df


def median_by_hour(df) -> pd.DataFrame:
    df = df.groupby(by=['Year', 'Month', 'Day', 'Hour']).median().reset_index()
    df['Hour'] = df['Hour'] + 1

    return df


def add_daytype(df) -> pd.DataFrame:
    # Add day type column equivalent to citylearn
    # Mo 2 Di 3 Mi 4 Do 5 Fr 6 Sa 7 So 1
    df['Datetime'] = pd.to_datetime(df[['Month', 'Day', 'Year']].astype(str).apply(' '.join, 1),
                                    format='%m %d %Y')

    df['Day Type'] = df['Datetime'].dt.dayofweek
    df['Day Type'] = df['Day Type'] + 2
    df.loc[df['Day Type'] == 8, ['Day Type']] = 1

    return df


def add_predictions(weather) -> pd.DataFrame:
    # First just real values
    # TODO: add noise or real predictions
    for variable in WEATHER_VARS:
        for pred_horizon in [6, 12, 24]:
            col_name = f'{pred_horizon}h Prediction {variable}'
            weather[col_name] = weather.loc[:, variable]
            weather[col_name] = weather[col_name].shift(-pred_horizon)

            # fill missing prediction values with the actual ones at the time the forecast addresses
            for i in weather[np.isnan(weather[col_name])].index:
                weather.loc[i, col_name] = weather.loc[i + pred_horizon - 24][variable]

    return weather


def correct_time_series_start(df) -> pd.DataFrame:
    # the building time series starts on Month=7, Hour=24, Day Type=7
    # followed by Month=8, Hour=1, Day Type=1
    # hence drop columns from the begnning of the year until the last hour on last saturday in july
    # and append them to the end
    first_row_idx = df.index[(df['Month'] == 7) &
                             (df['Hour'] == 24) &
                             (df['Day Type'] == 7)].tolist()[-1]
    idx = df.index.tolist()
    del idx[:first_row_idx]
    df = df.reindex(idx + list(range(first_row_idx)))

    return df


def read_fuel_data(load_dir) -> pd.DataFrame:
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

    fuel['Renewable Sources'] = fuel[['Hydro', 'Wind', 'Other Renewables']].sum(axis=1)
    fuel['Other'] = fuel[['Dual Fuel', 'Natural Gas', 'Nuclear', 'Other Fossil Fuels']].sum(axis=1)
    fuel['Fossil Share'] = fuel['Other']/(fuel['Renewable Sources']+fuel['Other'])
    fuel = fuel.drop(columns=['Hydro', 'Wind', 'Other Renewables', 'Dual Fuel', 'Natural Gas', 'Nuclear',
                                'Other Fossil Fuels'])

    return fuel


def adapt_price(load_path, save_path, fuel_mix, alpha=6) -> None:
    price = pd.read_csv(load_path)['Electricity Pricing [$]']
    fossil_share = fuel_mix['Fossil Share']

    new_price = price + alpha * fossil_share
    new_price.to_csv(save_path, index=False)

    print(f'File saved under {save_path}')


if __name__ == '__main__':
    weather_filepath = '../datasets/weather_ny_42.30_-74.37_2021.csv'
    weather_save_filepath = 'citylearn/data/nydata/weather.csv'
    preprocess_data(weather_filepath, weather_save_filepath, weather=True)

    fuel_mix_dirpath = '../datasets/fuel_mix_ny_2021'
    fuel_mix_save_filepath = 'citylearn/data/nydata/fuelmix.csv'
    fuel_mix = preprocess_data(fuel_mix_dirpath, fuel_mix_save_filepath, weather=False)

    price_filepath = 'citylearn/data/test/pricing.csv'
    price_save_filepath = 'citylearn/data/nydata/pricing.csv'
    adapt_price(price_filepath, price_save_filepath, fuel_mix)

