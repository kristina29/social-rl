import pandas as pd
import numpy as np

from citylearn.utilities import get_predictions

if __name__ == '__main__':
    filepath = 'citylearn/data/testenv/pricing.csv'
    pricing = pd.read_csv(filepath)
    pricing = pricing.join(pd.read_csv('citylearn/data/testenv/Building_1.csv'))
    pricing = pricing.drop(columns=['Month', 'Day Type',
                                    'Daylight Savings Status', 'Indoor Temperature [C]',
                                    'Average Unmet Cooling Setpoint Difference [C]',
                                    'Indoor Relative Humidity [%]', 'Equipment Electric Power [kWh]',
                                    'DHW Heating [kWh]', 'Cooling Load [kWh]', 'Heating Load [kWh]',
                                    'Solar Generation [W/kW]'])

    pricing.loc[pricing['Hour'].isin([1, 2, 3, 4, 5, 6, 7, 8]), 'Electricity Pricing [$]'] = 0.
    pricing.loc[pricing['Hour'].isin([9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                      24]), 'Electricity Pricing [$]'] = 300.

    predictions = get_predictions(pricing['Electricity Pricing [$]'])
    pricing['6h Prediction Electricity Pricing [$]'] = np.array(predictions[6], dtype=float)
    pricing['12h Prediction Electricity Pricing [$]'] = np.array(predictions[12], dtype=float)
    pricing['24h Prediction Electricity Pricing [$]'] = np.array(predictions[24], dtype=float)

    pricing = pricing.drop(columns='Hour')
    pricing.to_csv(filepath, index=False)
