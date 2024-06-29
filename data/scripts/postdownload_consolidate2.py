import os 

from download import get_date_list

def check_files():
    """
    Delete the dates that are not complete (having all the hours)
    """
    date_list = get_date_list('2022-04-04', '2022-04-25')
    for date in date_list:
        hours = range(24) # 0-23
        hours = [str(hour).zfill(2) for hour in hours] # 00-23

        markedForDeletion = False 


        for hour in hours:
            filename = f'states_{date}-{hour}.csv.gz'
            if not os.path.exists(f'/workspace/deepflow/data/osstate/extracted/{filename}'):
                markedForDeletion = True
                break

        if markedForDeletion:
            os.system(f'rm -rf /workspace/deepflow/data/osstate/extracted/{date}*')
            print(f"Deleted {date} files")

check_files()