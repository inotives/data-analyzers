
import pathlib
from dotenv import dotenv_values

config = dotenv_values('.env')

ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = f"{ROOT_DIR}/data"

POSTGRES_USER = config['POSTGRES_USER'] 
POSTGRES_PWD = config['POSTGRES_PWD']
POSTGRES_PORT = config['POSTGRES_PORT'] 
POSTGRES_DB_NAME = config['POSTGRES_DBNAME']
POSTGRES_HOST = config['POSTGRES_HOST']
POSTGRES_DB_URL = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PWD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB_NAME}"
ETHERSCAN_APIKEY = config['ETHERSCAN_APIKEY']