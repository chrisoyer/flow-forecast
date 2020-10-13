from datetime import datetime
from typing import Callable, Dict, List, Tuple, Type
import argparse
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import json
from flood_forecast.model_dict_function import decoding_functions, pytorch_criterion_dict
from flood_forecast.preprocessing.pytorch_loaders import CSVTestLoader
from flood_forecast.time_model import TimeSeriesModel
from flood_forecast.utils import flatten_list_function

model = torch.nn.Module()
    
def getInference(self, params: Dict, path: str=None) -> pd.DataFrame:
    if params['gcs']==False and path is None:
        raise Exception("Path to the model not provided. Provide the path of the model, or set 'GCS' to True and link the library to GCS")
    elif params['gcs']:
        # wget model and pass path
        path = 'path.pth'
    model.load_state_dict(torch.load(path))

def main():
    """
    Main function which is called from the command line. Entrypoint for all ML models.
    """
    parser = argparse.ArgumentParser(description="Argument parsing for training and eval")
    parser.add_argument("-p", "--params", help="Path to model config file")
    args = parser.parse_args()
    with open(args.params) as f:
        config = json.load(f)
    model_type = config['model_type']
    # evaluate_model(trained_model)
    print("Process is now complete.")

if __name__ == "__main__":
    main()
