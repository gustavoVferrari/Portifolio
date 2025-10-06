from kaggle.api.kaggle_api_extended import KaggleApi
import os
# Autenticar
api = KaggleApi()
api.authenticate()

path = r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio\Classification\titanic\dados_titanic\raw'
# Fazer download dos arquivos da competição 'titanic'
api.competition_download_files('titanic', path=path)

print("Download concluído!")

import zipfile

print('descomnpactar dados')

path_zip = os.path.join(path, 'titanic.zip')
with zipfile.ZipFile(path_zip, 'r') as zip_ref:
    zip_ref.extractall(path)    
print('dados descompactados')