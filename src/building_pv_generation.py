import pandas as pd

import pvlib
from pvlib import location
from pvlib.irradiance import get_total_irradiance
from pvlib.pvarray import pvefficiency_adr

########################################################################################################################
##### Based on https://pvlib-python.readthedocs.io/en/latest/gallery/adr-pvarray/plot_simulate_fast.html
########################################################################################################################

df_raw = pd.read_csv('../datasets/building_data_test/weather_ny.csv')

df = pd.DataFrame({'dhi': df_raw['Diffuse Solar Radiation [W/m2]'],
                   'dni': df_raw['Direct Solar Radiation [W/m2]'],
                   'ghi': df_raw['GHI'],
                   'temp_air': df_raw['Outdoor Drybulb Temperature [C]'],
                   'wind_speed': df_raw['Wind Speed [m/s]'],
                   })

hour = list(df_raw['Hour'])
hour = [h if h <= 23 else 0 for h in hour]
hour = [pd.to_datetime(f'1900-01-01 {int(h)}:00:00', format='%Y-%m-%d %H:%M:%S') for h in hour]
df.index = hour
df.index = df.index + pd.Timedelta(minutes=30)

loc = location.Location(latitude=42.3, longitude=-74.37)
solpos = loc.get_solarposition(df.index)

TILT = 42.3
ORIENT = 180

total_irrad = get_total_irradiance(TILT, ORIENT,
                                   solpos.apparent_zenith, solpos.azimuth,
                                   df.dni, df.ghi, df.dhi)

df['poa_global'] = total_irrad.poa_global

df['temp_pv'] = pvlib.temperature.faiman(df.poa_global, df.temp_air,
                                         df.wind_speed)

adr_params = {'k_a': 0.99924,
              'k_d': -5.49097,
              'tc_d': 0.01918,
              'k_rs': 0.06999,
              'k_rsh': 0.26144
              }

df['eta_rel'] = pvefficiency_adr(df['poa_global'], df['temp_pv'], **adr_params)

for b_id in range(1, 18):
    building_data = pd.read_csv(f'citylearn/data/nydata/Building_{b_id}.csv')

    # Set the desired array size:
    P_STC = building_data['Solar Generation [W/kW]'].max() + 500 # (W)
    print(b_id, P_STC)

    # and the irradiance level needed to achieve this output:
    G_STC = 1000.  # (W/m2)

    building_data['Solar Generation [W/kW]'] = list(P_STC * df['eta_rel'] * (df['poa_global'] / G_STC))
    building_data.to_csv(f'citylearn/data/nydata_new_buildings2/Building_{b_id}.csv', index=False)
