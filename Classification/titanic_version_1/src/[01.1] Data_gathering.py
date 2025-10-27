# read data from kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import yaml
import zipfile

# Authetication
api = KaggleApi()
api.authenticate()

# ymal config load
yaml_path = r"C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio\Classification\titanic_version_1\src\config.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


def Data_gatheting(**params):

    print("Begin Download!")
    api.competition_download_files(
        params['competition'], 
        path=params['data_gathering_path'])
    print("Download completed!")
    

    print('unzip data')
    path_zip = os.path.join(
        params['data_gathering_path'], 
        f'{params['competition']}.zip')
    
    with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        zip_ref.extractall(params['data_gathering_path'])    
    print('data unziped')
    
    
if __name__ == "__main__":
    
    params = {
        'competition': config['data_gathering']['competition'],
        'data_gathering_path': config['data_gathering']['path']
    }
    
    Data_gatheting(**params)