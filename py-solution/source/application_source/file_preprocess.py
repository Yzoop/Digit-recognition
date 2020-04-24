from easygui import fileopenbox
from source.data_management import data_manager

def get_data_from_file():
    try:
        file_path = fileopenbox()
        return data_manager.get_local_training_data(file_path)
    except:
        print('can not open file')