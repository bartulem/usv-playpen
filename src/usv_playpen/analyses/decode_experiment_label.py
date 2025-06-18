"""
@author: bartulem
Extracts information about what kind of experiment was conducted.
"""

from __future__ import annotations

import re


def extract_information(experiment_code: str | None = None) -> dict | None | None:
    """
    Extracts information about the experiment from the experiment code, as follows:

    A - ablation
    E - ephys
    H - chemogenetics
    O - optogenetics
    P - playback
    B - behavior
    V - devocalization
    U - urine/bedding

    Q - alone
    C - courtship
    X - females
    Y - males

    L - light
    D - dark

    1,2,3 ... - number of animals

    F - female
    M - male

    S - single
    G - group

    p - proestrus
    e - estrus
    m - matestrus
    d - diestrus


    Parameters
    ----------
    experiment_code (str)
        Code that describes the experiment, defaults to None.

    Returns
    -------
    output_dict (dict)
        Contains information about the experiment:
        experiment type, mouse number, mouse sex, mouse housing and mouse estrus.
    """

    if experiment_code is not None and type(experiment_code) == str:
        decoding_dict = {
            "A": "ablation",
            "E": "ephys",
            "H": "chemogenetics",
            "O": "optogenetics",
            "P": "playback",
            "B": "behavior",
            "V": "devocalization",
            "U": "urine/bedding",
            "Q": "alone",
            "C": "courtship",
            "X": "females",
            "Y": "males",
            "L": "light",
            "D": "dark",
            "F": "female",
            "M": "male",
            "S": "single",
            "G": "group",
            "p": "proestrus",
            "e": "estrus",
            "m": "matestrus",
            "d": "diestrus",
        }

        search_patterns_dict = {
            "experiment_type": r"[AEHOPBVUQCXYLD]",
            "mouse_number": r"\d",
            "mouse_sex": r"[MF]",
            "mouse_housing": r"[SG]",
            "mouse_estrus": r"[pemd]",
        }

        output_dict = {
            "experiment_type": [],
            "mouse_number": 0,
            "mouse_sex": [],
            "mouse_housing": [],
            "mouse_estrus": [],
        }

        for key in output_dict:
            if key != "mouse_number":
                for item in re.findall(search_patterns_dict[key], experiment_code):
                    output_dict[key].append(decoding_dict[item])
            else:
                output_dict[key] = int(
                    re.search(search_patterns_dict[key], experiment_code).group(0)
                )

        return output_dict

    return None
