# Import necessary libraries
import pandas as pd
import numpy as np

# Load data extraction and wrangling scripts
from src.spacex_falcon_9_data_extraction_using_api import extract_data
from src.spacex_data_wrangling import wrangle_data
from src.analysis_and_prediction import analyze_and_predict

def main():
    # Extract data
    raw_data = extract_data()
    
    # Wrangle data
    clean_data = wrangle_data(raw_data)
    
    # Analyze and predict
    analyze_and_predict(clean_data)

if __name__ == '__main__':
    main()