import pandas as pd

WEATHER_VARS = ['Outdoor Drybulb Temperature [C]',
                'Relative Humidity [%]',
                'Diffuse Solar Radiation [W/m2]',
                'Direct Solar Radiation [W/m2]',
                'Wind Speed [m/s]']

FINAL_ORDER = ['Outdoor Drybulb Temperature [C]',
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
               '24h Prediction Direct Solar Radiation [W/m2]',
               'Wind Speed [m/s]',
               '6h Prediction Wind Speed [m/s]',
               '12h Prediction Wind Speed [m/s]',
               '24h Prediction Wind Speed [m/s]']


def preprocess(load_path, save_path) -> None:
    weather = pd.read_csv(load_path, skiprows=2)

    # rename columns to match citylearn framework
    weather = weather.rename(columns={'Temperature': WEATHER_VARS[0],
                                      'Relative Humidity': WEATHER_VARS[1],
                                      'DHI': WEATHER_VARS[2],
                                      'DNI': WEATHER_VARS[3],
                                      'Wind Speed': WEATHER_VARS[4]})

    weather = median_by_hour(weather)
    weather = add_daytype(weather)
    weather = add_predictions(weather)
    weather = correct_time_series_start(weather)

    # drop not needed columns
    weather = weather.drop(columns=['Datetime', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Day Type'])

    # reorder to match citylearn
    weather = weather[FINAL_ORDER]

    weather.to_csv(save_path, index=False)
    print(f'File saved under {save_path}')


def median_by_hour(weather) -> pd.DataFrame:
    weather = weather.groupby(by=['Year', 'Month', 'Day', 'Hour']).median().reset_index()
    weather['Hour'] = weather['Hour'] + 1

    return weather


def add_daytype(weather) -> pd.DataFrame:
    # Add day type column equivalent to citylearn
    # Mo 2 Di 3 Mi 4 Do 5 Fr 6 Sa 7 So 1
    weather['Datetime'] = pd.to_datetime(weather[['Month', 'Day', 'Year']].astype(str).apply(' '.join, 1),
                                         format='%m %d %Y')

    weather['Day Type'] = weather['Datetime'].dt.dayofweek
    weather['Day Type'] = weather['Day Type'] + 2
    weather.loc[weather['Day Type'] == 8, ['Day Type']] = 1

    return weather


def add_predictions(weather) -> pd.DataFrame:
    # First just real values
    # TODO: add noise or real predictions
    for variable in WEATHER_VARS:
        for pred_horizon in [6, 12, 24]:
            col_name = f'{pred_horizon}h Prediction {variable}'
            weather[col_name] = weather.loc[:, variable]
            weather[col_name] = weather[col_name].shift(-pred_horizon)

    return weather


def correct_time_series_start(weather) -> pd.DataFrame:
    # the building time series starts on Month=7, Hour=24, Day Type=7
    # followed by Month=8, Hour=1, Day Type=1
    # hence drop columns from the begnning of the year until the last hour on last saturday in july
    # and append them to the end
    first_row_idx = weather.index[(weather['Month'] == 7) &
                                  (weather['Hour'] == 24) &
                                  (weather['Day Type'] == 7)].tolist()[-1]
    idx = weather.index.tolist()
    del idx[:first_row_idx]
    weather = weather.reindex(idx + list(range(first_row_idx)))

    return weather


if __name__ == '__main__':
    weather_filepath = '../datasets/weather_ny_42.30_-74.37_2021.csv'
    save_filepath = 'citylearn/data/nydata/weather.csv'
    preprocess(weather_filepath, save_filepath)
