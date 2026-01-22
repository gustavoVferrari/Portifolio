import sys
sys.path.append(r"Classification\titanic\model_log_reg")
import tqdm
import subprocess
import os


SCRIPTS_TO_RUN = [
    '[02.1] Feature_eng.py',    
    '[03.1] Model Selection.py',
    '[03.2] Train Model.py',
    '[04.1] Predict.py',
    '[05.2] Submission.py'        
]

path = r'C:\Users\gustavo\Documents\Data Science\08-GitHub\Portifolio\Classification\titanic\model_log_reg\src' 

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