import os 
from download import get_date_list

def make_dir() -> None:
    os.makedirs('/workspace/deepflow/data/osstate/extracted', exist_ok=True)

make_dir()

def extract_files() -> None:
    date_list = get_date_list('2022-04-04', '2022-04-25')
    # Get all the files in the directory
    files = os.listdir('/workspace/deepflow/data/osstate')
    print(f'Found {len(files)} files')
    # Only keep the files that contain the date in the date_list
    files = [file for file in files if any(date in file for date in date_list)]
    print(f'Will process {len(files)} files due to date_list')
    # Extract all the files
    for file in files:
        os.system(f'tar -xvf /workspace/deepflow/data/osstate/{file} -C /workspace/deepflow/data/osstate/extracted')
        # os.system(f'rm data/osstate/{file}')

if __name__ == '__main__':
    extract_files()