import os
from pathlib import Path
import shutil
from typing import Iterable, List, Union
import numpy as np

from citylearn.utilities import read_json, get_predictions


class DataSet:
    __ROOT_DIRECTORY = os.path.join(os.path.dirname(__file__),'data')

    @staticmethod
    def get_names() -> List[str]:
        return sorted([
            d for d in os.listdir(DataSet.__ROOT_DIRECTORY) 
            if os.path.isdir(os.path.join(DataSet.__ROOT_DIRECTORY,d))
        ])

    @staticmethod
    def copy(name: str, destination_directory: Union[Path, str] = None):
        source_directory = os.path.join(DataSet.__ROOT_DIRECTORY,name)
        destination_directory = '' if destination_directory is None else destination_directory
        destination_directory = os.path.join(destination_directory,name)
        os.makedirs(destination_directory,exist_ok=True)

        for f in os.listdir(source_directory):
            if f.endswith('.csv') or f.endswith('.json'):
                source_filepath = os.path.join(source_directory,f)
                destination_filepath = os.path.join(destination_directory,f)
                shutil.copy(source_filepath,destination_filepath)
            else:
                continue

    @staticmethod
    def get_schema(name: str):
        root_directory = os.path.join(DataSet.__ROOT_DIRECTORY,name)
        filepath = os.path.join(root_directory,'schema.json')
        schema = read_json(filepath)
        schema['root_directory'] = root_directory
        return schema

class EnergySimulation:
    """`Building` `energy_simulation` data class.

    Attributes
    ----------
    month : np.array
        Month time series value ranging from 1 - 12.
    hour : np.array
        Hour time series value ranging from 1 - 24.
    day_type : np.array
        Numeric day of week time series ranging from 1 - 8 where 1 - 7 is Monday - Sunday and 8 is reserved for special days e.g. holiday.
    daylight_savings_status : np.array
        Daylight saving status time series signal of 0 or 1 indicating inactive  or active daylight saving respectively.
    indoor_dry_bulb_temperature : np.array
        Zone volume-weighted average building dry bulb temperature time series in [C].
    average_unmet_cooling_setpoint_difference : np.array
        Zone volume-weighted average difference between `indoor_dry_bulb_temperature` and cooling temperature setpoints time series in [C].
    indoor_relative_humidity : np.array
        Zone volume-weighted average building relative humidity time series in [%].
    non_shiftable_load : np.array
        Total building non-shiftable plug and equipment loads time series in [kWh].
    dhw_demand : np.array
        Total building domestic hot water demand time series in [kWh].
    cooling_demand : np.array
        Total building space cooling demand time series in [kWh].
    heating_demand : np.array
        Total building space heating demand time series in [kWh].
    solar_generation : np.array
        Inverter output per 1 kW of PV system time series in [W/kW].
    """

    def __init__(
        self, month: Iterable[int], hour: Iterable[int], day_type: Iterable[int],
        daylight_savings_status: Iterable[int], indoor_dry_bulb_temperature: Iterable[float], average_unmet_cooling_setpoint_difference: Iterable[float], indoor_relative_humidity: Iterable[float], 
        non_shiftable_load: Iterable[float], dhw_demand: Iterable[float], cooling_demand: Iterable[float], heating_demand: Iterable[float],
        solar_generation: Iterable[float]
    ):
        r"""Initialize `EnergySimulation`."""

        self.month = np.array(month, dtype = int)
        self.hour = np.array(hour, dtype = int)
        self.day_type = np.array(day_type, dtype = int)
        self.daylight_savings_status = np.array(daylight_savings_status, dtype = int)
        self.indoor_dry_bulb_temperature = np.array(indoor_dry_bulb_temperature, dtype = float)
        self.average_unmet_cooling_setpoint_difference = np.array(average_unmet_cooling_setpoint_difference, dtype = float)
        self.indoor_relative_humidity = np.array(indoor_relative_humidity, dtype = float)
        self.non_shiftable_load = np.array(non_shiftable_load, dtype = float)
        self.dhw_demand = np.array(dhw_demand, dtype = float)
        self.cooling_demand = np.array(cooling_demand, dtype = float)
        self.heating_demand = np.array(heating_demand, dtype = float)
        self.solar_generation = np.array(solar_generation, dtype = float)

class Weather:
    """`Building` `weather` data class.

    Attributes
    ----------
    outdoor_dry_bulb_temperature : np.array
        Outdoor dry bulb temperature time series in [C].
    outdoor_relative_humidity : np.array
        Outdoor relative humidity time series in [%].
    diffuse_solar_irradiance : np.array
        Diffuse solar irradiance time series in [W/m^2].
    direct_solar_irradiance : np.array
        Direct solar irradiance time series in [W/m^2].
    outdoor_dry_bulb_temperature_predicted_6h : np.array
        Outdoor dry bulb temperature 6 hours ahead prediction time series in [C].
    outdoor_dry_bulb_temperature_predicted_12h : np.array
        Outdoor dry bulb temperature 12 hours ahead prediction time series in [C].
    outdoor_dry_bulb_temperature_predicted_24h : np.array
        Outdoor dry bulb temperature 24 hours ahead prediction time series in [C].
    outdoor_relative_humidity_predicted_6h : np.array
        Outdoor relative humidity 6 hours ahead prediction time series in [%].
    outdoor_relative_humidity_predicted_12h : np.array
        Outdoor relative humidity 12 hours ahead prediction time series in [%].
    outdoor_relative_humidity_predicted_24h : np.array
        Outdoor relative humidity 24 hours ahead prediction time series in [%].
    diffuse_solar_irradiance_predicted_6h : np.array
        Diffuse solar irradiance 6 hours ahead prediction time series in [W/m^2].
    diffuse_solar_irradiance_predicted_12h : np.array
        Diffuse solar irradiance 12 hours ahead prediction time series in [W/m^2].
    diffuse_solar_irradiance_predicted_24h : np.array
        Diffuse solar irradiance 24 hours ahead prediction time series in [W/m^2].
    direct_solar_irradiance_predicted_6h : np.array
        Direct solar irradiance 6 hours ahead prediction time series in [W/m^2].
    direct_solar_irradiance_predicted_12h : np.array
        Direct solar irradiance 12 hours ahead prediction time series in [W/m^2].
    direct_solar_irradiance_predicted_24h : np.array
        Direct solar irradiance 24 hours ahead prediction time series in [W/m^2].
    """

    def __init__(
        self, outdoor_dry_bulb_temperature: Iterable[float], outdoor_relative_humidity: Iterable[float], diffuse_solar_irradiance: Iterable[float], direct_solar_irradiance: Iterable[float],
        outdoor_dry_bulb_temperature_predicted_6h: Iterable[float], outdoor_dry_bulb_temperature_predicted_12h: Iterable[float], outdoor_dry_bulb_temperature_predicted_24h: Iterable[float],
        outdoor_relative_humidity_predicted_6h: Iterable[float], outdoor_relative_humidity_predicted_12h: Iterable[float], outdoor_relative_humidity_predicted_24h: Iterable[float],
        diffuse_solar_irradiance_predicted_6h: Iterable[float], diffuse_solar_irradiance_predicted_12h: Iterable[float], diffuse_solar_irradiance_predicted_24h: Iterable[float],
        direct_solar_irradiance_predicted_6h: Iterable[float], direct_solar_irradiance_predicted_12h: Iterable[float], direct_solar_irradiance_predicted_24h: Iterable[float],
    ):
        r"""Initialize `Weather`."""

        self.outdoor_dry_bulb_temperature = np.array(outdoor_dry_bulb_temperature, dtype = float)
        self.outdoor_relative_humidity = np.array(outdoor_relative_humidity, dtype = float)
        self.diffuse_solar_irradiance = np.array(diffuse_solar_irradiance, dtype = float)
        self.direct_solar_irradiance = np.array(direct_solar_irradiance, dtype = float)
        self.outdoor_dry_bulb_temperature_predicted_6h = np.array(outdoor_dry_bulb_temperature_predicted_6h, dtype = float)
        self.outdoor_dry_bulb_temperature_predicted_12h = np.array(outdoor_dry_bulb_temperature_predicted_12h, dtype = float)
        self.outdoor_dry_bulb_temperature_predicted_24h = np.array(outdoor_dry_bulb_temperature_predicted_24h, dtype = float)
        self.outdoor_relative_humidity_predicted_6h = np.array(outdoor_relative_humidity_predicted_6h, dtype = float)
        self.outdoor_relative_humidity_predicted_12h = np.array(outdoor_relative_humidity_predicted_12h, dtype = float)
        self.outdoor_relative_humidity_predicted_24h = np.array(outdoor_relative_humidity_predicted_24h, dtype = float)
        self.diffuse_solar_irradiance_predicted_6h = np.array(diffuse_solar_irradiance_predicted_6h, dtype = float)
        self.diffuse_solar_irradiance_predicted_12h = np.array(diffuse_solar_irradiance_predicted_12h, dtype = float)
        self.diffuse_solar_irradiance_predicted_24h = np.array(diffuse_solar_irradiance_predicted_24h, dtype = float)
        self.direct_solar_irradiance_predicted_6h = np.array(direct_solar_irradiance_predicted_6h, dtype = float)
        self.direct_solar_irradiance_predicted_12h = np.array(direct_solar_irradiance_predicted_12h, dtype = float)
        self.direct_solar_irradiance_predicted_24h = np.array(direct_solar_irradiance_predicted_24h, dtype = float)

class WeatherWind(Weather):
    """`Building` `wind weather` data class.

    Attributes
    ----------
    wind_speed : np.array, optional
        Wind speed time series in [m/s].
    wind_speed_predicted_6h : np.array, optional
        Wind speed 6 hours ahead prediction time series in [m/s].
    wind_speed_predicted_12h : np.array, optional
        Wind speed 12 hours ahead prediction time series in [m/s].
    wind_speed_predicted_24h : np.array, optional
        Wind speed 24 hours ahead prediction time series in [m/s].
    """

    def __init__(
        self, wind_speed: Iterable[float], wind_speed_predicted_6h: Iterable[float],
        wind_speed_predicted_12h: Iterable[float], wind_speed_predicted_24h: Iterable[float], *args,
    ):
        r"""Initialize `WeatherWind`.

        Parameters
        ----------
        *args : tuple
            `Weather` positional arguments.

        """

        super().__init__(*args)

        self.wind_speed = np.array(wind_speed, dtype=float)
        self.wind_speed_predicted_6h = np.array(wind_speed_predicted_6h, dtype=float)
        self.wind_speed_predicted_12h = np.array(wind_speed_predicted_12h, dtype=float)
        self.wind_speed_predicted_24h = np.array(wind_speed_predicted_24h, dtype=float)

class Pricing:
    """`Building` `pricing` data class.

    Attributes
    ----------
    electricity_pricing : np.array
        Electricity pricing time series in [$].
    electricity_pricing_predicted_6h : np.array
        Electricity pricing 6 hours ahead prediction time series in [$].
    electricity_pricing_predicted_12h : np.array
        Electricity pricing 12 hours ahead prediction time series in [$].
    electricity_pricing_predicted_24h : np.array
        Electricity pricing 24 hours ahead prediction time series in [$].
    """

    def __init__(
        self, electricity_pricing: Iterable[float], electricity_pricing_predicted_6h: Iterable[float], 
        electricity_pricing_predicted_12h: Iterable[float], electricity_pricing_predicted_24h: Iterable[float]
    ):
        r"""Initialize `Pricing`."""

        self.electricity_pricing = np.array(electricity_pricing, dtype = float)
        self.electricity_pricing_predicted_6h = np.array(electricity_pricing_predicted_6h, dtype = float)
        self.electricity_pricing_predicted_12h = np.array(electricity_pricing_predicted_12h, dtype = float)
        self.electricity_pricing_predicted_24h = np.array(electricity_pricing_predicted_24h, dtype = float)

    def weight_by_fossil_share(self, pricing_weight_fossil: float, renewable_share: Iterable[float]):
        """Scale the eletricity price by the share of fossil energy by factor `pricing_weight_fossil`.
            The higher the share of fossil energy produced, the higher the price.

            Parameters
            ----------
            pricing_weight_fossil: float
                Factor by which the price should be scaled.
            renewable_share: Iterable[float]
                Time series of the renewable energy production share.
        """

        fossil_share = 1-renewable_share

        self.electricity_pricing = self.electricity_pricing + pricing_weight_fossil * fossil_share

        # set minimum price to 0.2
        self.electricity_pricing = self.electricity_pricing - self.electricity_pricing.min() + 0.2

        predictions = get_predictions(self.electricity_pricing)
        self.electricity_pricing_predicted_6h = np.array(predictions[6], dtype=float)
        self.electricity_pricing_predicted_12h = np.array(predictions[12], dtype=float)
        self.electricity_pricing_predicted_24h = np.array(predictions[24], dtype=float)


class CarbonIntensity:
    """`Building` `carbon_intensity` data class.

    Attributes
    ----------
    carbon_intensity : np.array
        Grid carbon emission rate time series in [kg_co2/kWh].
    """

    def __init__(self, carbon_intensity: Iterable[float]):
        r"""Initialize `CarbonIntensity`."""

        self.carbon_intensity = np.array(carbon_intensity, dtype = float)

class FuelMix:
    """`Building` `fuel_mix` data class.

    Attributes
    ----------
    renewable_energy_produced : np.array
        Renewable energy production time series in [kWh].
    non_renewable_energy_produced : np.array
        Non-Renewable energy production prediction time series in [kWh].
    renewable_energy_share: np.array
        Renewable energy share of total energy produced.
    """

    def __init__(self, renewable_energy_produced: Iterable[float], non_renewable_energy_produced: Iterable[float],
                 renewable_energy_share: Iterable[float]):
        r"""Initialize `FuelMix`."""

        self.renewable_energy_produced = np.array(renewable_energy_produced, dtype = float)
        self.non_renewable_energy_produced = np.array(non_renewable_energy_produced, dtype=float)
        self.renewable_energy_share = np.array(renewable_energy_share, dtype=float)

    def scale_to_buildings(self, sum_medians: float) -> None:
        """Scale the renewable and non-renewable energy production to
            the sum of the median consumed energy of all included buildings.

            Parameters
            ----------
            sum_medians: float
                Sum of the medians of net energy consumption of all included buildings.
        """
        self.renewable_energy_produced = self.renewable_energy_share * sum_medians
        self.non_renewable_energy_produced = (1 - self.renewable_energy_share) * sum_medians
