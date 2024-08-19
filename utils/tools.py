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
