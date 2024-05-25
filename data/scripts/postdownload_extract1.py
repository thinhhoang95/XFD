import os 

def extract_files() -> None:
    # Get all the files in the directory
    files = os.listdir('data/osstate')
    # Extract all the files
    for file in files:
        os.system(f'tar -xvf data/osstate/{file} -C data/osstate/extracted')
        # os.system(f'rm data/osstate/{file}')

if __name__ == '__main__':
    extract_files()