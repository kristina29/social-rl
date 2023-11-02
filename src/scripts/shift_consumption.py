import random
from optparse import OptionParser, Option

import pandas as pd
import re

optParser = OptionParser(option_class=Option)
optParser.add_option('--building', action='store', dest='building_path',
                     default='citylearn/data/nnblo1_onlyb3shifted/Building_3.csv')
optParser.add_option('--median', action='store', type='int', dest='median', default=0.25)
optParser.add_option('-n', action='store', type='int', dest='n', default=5)
optParser.add_option('--r', action='store', type='int', dest='r', default=1)
optParser.add_option('--output', action='store', dest='output_path', default='citylearn/data/nnblo1_onlyb3shifted/')

opts, args = optParser.parse_args()

if __name__ == '__main__':
    building_path = opts.building_path
    output_path = opts.output_path
    median = opts.median
    n = opts.n
    r = opts.r

    building_df = pd.read_csv(building_path)
    demand = building_df['Equipment Electric Power [kWh]']
    solar = building_df['Solar Generation [W/kW]']

    b_id = int(re.findall(r'\d+', building_path.split('/')[-1])[0])
    id = 1
    for i in range(n):
        shift_by = round(random.uniform(0.2, r), 2)

        demand_shifted = demand + shift_by
        solar_shifted = solar + shift_by

        new_df = building_df.copy(deep=True)
        new_df['Equipment Electric Power [kWh]'] = demand_shifted
        new_df['Solar Generation [W/kW]'] = solar_shifted
        new_median = median + shift_by

        if id == b_id:
            id += 1

        path = f'{output_path}Building_{id}.csv'
        #new_df.to_csv(path, index=False)
        #print(f'Written to {path} --- Shifted by: {shift_by}, Median: {new_median}')
        id += 1

