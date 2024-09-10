import pandas as pd 
import pathlib

ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = f"{ROOT_DIR}/data"

def load_csv_from_data(csv_file):
    """Importing Data from CSV """    
    file_path = f"{DATA_DIR}/{csv_file}.csv"

    return pd.read_csv(file_path)


def export_data_to_csv(data, filename):
    """Exporting Data to CSV """
    exported_dir = f"{DATA_DIR}/_OUTPUTS/{filename}.csv"
    
    data.to_csv(exported_dir, index=False)
    
    print(f">> Data exported to :: {exported_dir}")

    return 

def data_load_ohlcv(csvfile, start=None, end=None):
    """Load Data from CSV to Dataframe"""
    csv_path = f"{DATA_DIR}/{csvfile}.csv"
    data = pd.read_csv(csv_path, sep=';', parse_dates=['timeOpen', 'timeClose', 'timeHigh', 'timeLow', 'timestamp'])
    data = data.rename(columns={'timeOpen': 'date'})

    # extract only ohlcv data and date.
    ohlcv = data[['date', 'open', 'high', 'low', 'close', 'volume', 'marketCap']].sort_values('date').drop_duplicates(subset=['date']).reset_index(drop=True)

    # Set date as index.
    ohlcv.set_index('date', inplace=True)

    return ohlcv[start:end]