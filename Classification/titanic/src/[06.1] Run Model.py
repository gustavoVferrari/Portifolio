import sys
sys.path.append(r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio/Classification/titanic/src')
import tqdm
import subprocess
import os

path = r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio/Classification/titanic/src/' 


SCRIPTS_TO_RUN = [
    '[01.1] Data_gathering.py',
    '[01.2] Feature_creation.py',
    '[02.1] Feature_eng.py',
    '[02.2] Feature_selection.py',
    '[02.3] Model Selection.py',
    '[03.1] Train Model.py',
    '[04.1] Predict.py'    
]

def run_script_safely(path, script_path):    
    result = subprocess.run(
                [sys.executable, os.path.join(path, script_path)],
                capture_output=True,
                text=True,
                check=True
            )
    
if __name__ == '__main__':
    
    for script in tqdm.tqdm(SCRIPTS_TO_RUN):
        print(f'Run script {script}')
        run_script_safely(path, script)