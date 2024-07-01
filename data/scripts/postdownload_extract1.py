import os 
from download import get_date_list

from path_prefix import PATH_PREFIX

def make_dir() -> None:
    os.makedirs(f'{PATH_PREFIX}/data/osstate/extracted', exist_ok=True)
    print('Created extracted directory', f'{PATH_PREFIX}/data/osstate/extracted')

make_dir()

def extract_files() -> None:
    date_list = get_date_list('2022-04-04', '2022-06-27')
    date_list = date_list[:1] # Only process the first date for testing
    # Get all the files in the directory
    files = os.listdir(f'{PATH_PREFIX}/data/osstate')
    print(f'Found {len(files)} files')
    # Only keep the files that contain the date in the date_list
    files = [file for file in files if any(date in file for date in date_list)]
    print(f'Will process {len(files)} files due to date_list')
    # Extract all the files
    for file in files:
        os.system(f'tar -xvf {PATH_PREFIX}/data/osstate/{file} -C {PATH_PREFIX}/data/osstate/extracted')
        # os.system(f'rm data/osstate/{file}')

if __name__ == '__main__':
    extract_files()