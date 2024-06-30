import os 

from path_prefix import PATH_PREFIX

from download import get_date_list

def check_files():
    """
    Delete the dates that are not complete (having all the hours)
    """
    date_list = get_date_list('2022-04-04', '2022-04-25')
    filename_to_delete = []
    for date in date_list:
        hours = range(24) # 0-23
        hours = [str(hour).zfill(2) for hour in hours] # 00-23

        for hour in hours:
            filename = f'states_{date}-{hour}.csv.gz'
            if not os.path.exists(f'{PATH_PREFIX}/data/osstate/extracted/{filename}'):
                filename_to_delete.append(filename)

    print(f"Found {len(filename_to_delete)} files to delete")

    confirm = input("Are you sure you want to delete the files? (y/n): ")
    if confirm.lower() == "y":
        for filename in filename_to_delete:
            os.remove(f'{PATH_PREFIX}/data/osstate/extracted/{filename}')
            print(f"Deleted file: {filename}")
    else:
        print("Deletion cancelled.")

check_files()