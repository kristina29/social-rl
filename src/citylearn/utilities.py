from typing import List

import simplejson as json


def read_json(filepath: str, **kwargs):
    """Return json document as dictionary.
    
    Parameters
    ----------
    filepath : str
       pathname of JSON document.

    Other Parameters
    ----------------
    **kwargs : dict
        Other infrequently used keyword arguments to be parsed to `simplejson.load`.
       
    Returns
    -------
    dict
        JSON document converted to dictionary.
    """

    with open(filepath) as f:
        json_file = json.load(f, **kwargs)

    return json_file


def write_json(filepath: str, dictionary: dict, **kwargs):
    """Write dictionary to json file. Returns boolen value of operation success. 
    
    Parameters
    ----------
    filepath : str
        pathname of JSON document.
    dictionary: dict
        dictionary to convert to JSON.

    Other Parameters
    ----------------
    **kwargs : dict
        Other infrequently used keyword arguments to be parsed to `simplejson.dump`.
    """

    kwargs = {'ignore_nan': True, 'sort_keys': False, 'default': str, 'indent': 2, **kwargs}
    with open(filepath, 'w') as f:
        json.dump(dictionary, f, **kwargs)


def get_active_parts(schema: dict, key: str) -> List[str]:
    """Get objects of the schema with specified key that are active.

        Parameters
        ----------
        schema : dict
            Dictionary object of a CityLearn schema
        key: str
            key of the object type to filter the active ones (e.g. buildings, observations)
        """
    active_parts = []
    all_parts = schema[key]

    # parameter that defines if the object is active
    active_param = 'active'
    if key == 'buildings':  #
        active_param = 'include'

    for part in all_parts:
        active_parts.append(part) if all_parts[part][active_param] else None

    return active_parts
