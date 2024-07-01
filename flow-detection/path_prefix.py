
PATH_PREFIX = '/Users/thinhhoang/Documents/XFD'

def extract_file_name(file_path: str) -> str:
    # Return everything after the last '/' and before the first '.'
    if '/' in file_path:
        file_path = file_path.split('/')[-1]
    if '.' in file_path:
        file_path = file_path.split('.')[0]
    return file_path