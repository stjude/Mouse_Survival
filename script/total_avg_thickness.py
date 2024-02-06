import numpy as np
import nibabel as nib
import os
import pandas as pd
from glob import glob

def calculate_average_thickness(thickness_map):
    """
    Calculate the average cortical thickness for a 3D thickness map,
    excluding zeros from the calculation.
    """
    thickness_map[thickness_map == 0] = np.nan  # Convert zeros to NaN
    average_thickness = np.nanmean(thickness_map)  # Calculate mean, ignoring NaNs
    return average_thickness

def process_files(thickness_maps):
    """
    Process a list of thickness map files and return a DataFrame with average thickness for each mouse.
    """
    averages = []
    mice_names = []

    for file_path in thickness_maps:
        thickness_map = nib.load(file_path).get_fdata()
        average_thickness = calculate_average_thickness(thickness_map)
        averages.append(average_thickness)

        # Extract an identifier for each mouse from the file name
        mouse_name = os.path.basename(file_path).split('_')[0]
        mice_names.append(mouse_name)

    # Create a DataFrame with the collected averages
    df = pd.DataFrame({'Mouse': mice_names, 'Average Thickness': averages})
    return df

# Directory path containing thickness maps
directory_path = 'new_data\\nonsurvivor\\thickness_maps'

# Find all thickness map files in the directory
thickness_maps = glob(os.path.join(directory_path, '*thickness_map.nii.gz'))

# Process the files and get the DataFrame
thickness_averages_df = process_files(thickness_maps)

if not thickness_averages_df.empty:
    # Save the DataFrame to a CSV file
    output_csv_path = 'new_data\\nonsurvivor\\nonsurvivor_average_thickness_totals.csv'
    thickness_averages_df.to_csv(output_csv_path, index=False)
    print("Mouse thickness averages saved to:", output_csv_path)
else:
    print("No thickness maps were processed.")
