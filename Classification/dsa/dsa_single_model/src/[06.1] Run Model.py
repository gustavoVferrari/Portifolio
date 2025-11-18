import sys
sys.path.append(r'Classification/dsa/dsa_single_model')
import tqdm
import subprocess
import os

path = r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio\Classification\dsa\dsa_single_model\src' 


SCRIPTS_TO_RUN = [
    # '[01.1] Feature_creation.py',
    # '[01.2] Feature_analysis.py',
    # '[02.1] Feature_eng.py',
    '[02.2] Feature_selection.py',
    '[03.1] Model Selection.py',
    '[03.2] Train Model.py',
    '[04.1] Predict.py',
    '[05.2] Submission.py'        
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