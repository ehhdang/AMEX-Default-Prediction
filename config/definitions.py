import os 
from pathlib import Path

config_folder_directory = Path(os.path.dirname(os.path.realpath(__file__)))
DOTENV_PATH = config_folder_directory.joinpath(".env")
ROOT_DIRECTORY = config_folder_directory.parent.absolute()
CURRENT_WORKING_DIRECTORY = os.getcwd()