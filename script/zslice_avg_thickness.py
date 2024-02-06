import os
import numpy as np
import nibabel as nib
import pandas as pd

thickness_maps_dir = 'new_data\\nonsurvivor\\thickness_maps'
mice_files = [f for f in os.listdir(thickness_maps_dir) if f.endswith('.nii.gz')]  # List all .nii.gz files
slice_thickness_averages = {}  # Dictionary to hold slice averages for each mouse

for mouse_file in mice_files:
    # Load the thickness map for the current mouse
    print(f"Calculating avearages for {mouse_file}.")
    thickness_map_path = os.path.join(thickness_maps_dir, mouse_file)
    thickness_map_img = nib.load(thickness_map_path)
    thickness_map = thickness_map_img.get_fdata()

    # Create a masked array where zeros are masked out
    masked_thickness_map = np.ma.masked_where(thickness_map == 0, thickness_map)

    # Calculate the average thickness for each z-slice
    slice_averages = np.nanmean(masked_thickness_map, axis=(0, 1))  # Averages along x and y for each z
    slice_thickness_averages[mouse_file] = slice_averages.filled(np.nan)  # Replace masked values with NaN for consistency

# Convert the dictionary to a DataFrame
df = pd.DataFrame.from_dict(slice_thickness_averages, orient='index')

# Optionally, name the columns as z-slice positions
num_slices = df.shape[1]
df.columns = [f'Z-Slice {i+1}' for i in range(num_slices)]

# Name the index as 'Mouse'
df.index.name = 'Mouse'

output_csv_path = 'new_data\\nonsurvivor\\nonsurvivor_thickness_map_avgs.csv'
df.to_csv(output_csv_path)
