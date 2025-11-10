# read data from kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import yaml
import zipfile

# Authetication
api = KaggleApi()
api.authenticate()

# ymal config load
yaml_path = r"Classification\titanic\src\init_config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


def Data_gatheting(**params):

    print("Begin Download!")
    api.competition_download_files(
        params["competition"], 
        path=os.path.join(
            params["init_path"],
            params["path"],
            ))
    print("Download completed!")    

    print('unzip data')
    path_zip = os.path.join(
        params["init_path"],
        params["path"], 
        f'{params["competition"]}.zip')
    
    with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        zip_ref.extractall(
            os.path.join(
                params["init_path"],
                params["path"])
        )    
    print('data unziped')
    
    
if __name__ == "__main__":
    
    params = {
        'competition': config['data_gathering']['competition'],
        'init_path': config['data_gathering']['init_path'],
        'path': config['data_gathering']['path']
    }
    
    Data_gatheting(**params)