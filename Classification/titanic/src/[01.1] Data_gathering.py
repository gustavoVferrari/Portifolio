from kaggle.api.kaggle_api_extended import KaggleApi

# Autenticar
api = KaggleApi()
api.authenticate()

# Fazer download dos arquivos da competição 'titanic'
api.competition_download_files('titanic', path='../dados_titanic/raw')

print("Download concluído!")

import zipfile

print('descomnpactar dados')
with zipfile.ZipFile('../dados_titanic/raw/titanic.zip', 'r') as zip_ref:
    zip_ref.extractall('../dados_titanic/raw/')    
print('dados descompactados')