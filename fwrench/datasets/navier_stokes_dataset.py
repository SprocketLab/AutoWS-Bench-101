import os
import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd

def check_and_download():
    
    folder_path = "./datasets/navier_stokes_data"

    x_test_id = "1uCgMlkyomDTmnEWXoC78fQBwojT0xO2A"
    x_test_filename = "x_test.npy"

    x_val_id = "1v1DkSfarlfTvHEgGNc12_njCL2ns7mxY"
    x_val_filename = 'x_val.npy'

    y_test_id = "1Di0awy7oAL-MBZeoVEDc8jXcTUc8HS76"
    y_test_filename = 'y_test.npy'

    y_val_id = "1RqM52Jeogauf6lANYm2j9RRllQlcCMhJ"
    y_val_filename = 'y_val.npy'
    
    if os.path.exists(folder_path) == False:
        os.mkdir(folder_path, 0o777)

    x_test_filepath = os.path.join(folder_path, x_test_filename)
    if os.path.exists(x_test_filepath) == False:
        gdd.download_file_from_google_drive(file_id=x_test_id,
                                            dest_path=x_test_filepath,
                                            unzip=True)
    
    x_val_filepath = os.path.join(folder_path, x_val_filename)
    if os.path.exists(x_val_filepath) == False:
        gdd.download_file_from_google_drive(file_id=x_val_id,
                                            dest_path=x_val_filepath,
                                            unzip=True)
    
    y_test_filepath = os.path.join(folder_path, y_test_filename)
    if os.path.exists(y_test_filepath) == False:
        gdd.download_file_from_google_drive(file_id=y_test_id,
                                            dest_path=y_test_filepath,
                                            unzip=True)
        
    y_val_filepath = os.path.join(folder_path, y_val_filename)
    if os.path.exists(y_val_filepath) == False:
        gdd.download_file_from_google_drive(file_id=y_val_id,
                                            dest_path=y_val_filepath,
                                            unzip=True)
    
    valid_X_np = np.load(x_val_filepath)
    valid_y_np = np.load(y_val_filepath)
    train_X_np = np.load(x_val_filepath)
    train_y_np = np.load(y_val_filepath)
    test_X_np = np.load(x_test_filepath)
    test_y_np = np.load(y_test_filepath)
    
    return train_X_np, train_y_np, valid_X_np, valid_y_np, test_X_np, test_y_np