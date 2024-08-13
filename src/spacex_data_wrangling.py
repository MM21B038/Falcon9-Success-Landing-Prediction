import pandas as pd
import numpy as np

def wrangle_data(data):
    """## Making class column containing type of Result/Outcome"""
    landing_outcomes = data['Outcome'].value_counts()
    bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])
    bad_outcomes

    landing_class = []

    for key, value in data['Outcome'].items():
        if value in bad_outcomes:
            landing_class.append(0)
        else:
            landing_class.append(1)

    data['Class']=landing_class
    return data

    