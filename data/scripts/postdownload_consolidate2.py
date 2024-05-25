import os 

from download import get_date_list

def check_files():
    date_list = get_date_list('2022-01-03', '2022-01-31')
    for date in date_list:
        hours = range(24) # 0-23
        hours = [str(hour).zfill(2) for hour in hours] # 00-23

        markedForDeletion = False 


        for hour in hours:
            filename = f'states_{date}-{hour}.csv.gz'
            if not os.path.exists(f'data/osstate/extracted/{filename}'):
                markedForDeletion = True
                break

        if markedForDeletion:
            os.system(f'rm -rf data/osstate/extracted/{date}*')
            print(f"Deleted {date} files")

check_files()