import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, binary_dilation
import os

# Locate folder with cortex masks in zipped NIfTI format
cortex_dir = 'new_data\\nonsurvivor\\cortex_mask'

# Locate folder with whole brain masks in zipped NIfTI format
brain_dir = 'new_data\\nonsurvivor\\brain_mask'

# Load filenames from each folder
cortex_files = sorted([f for f in os.listdir(cortex_dir) if f.endswith('.nii.gz')])
brain_files = sorted([f for f in os.listdir(brain_dir) if f.endswith('.nii.gz')])

wm_dict = {}
pial_dict = {}

# Paths for saving the results
wm_output_dir = 'new_data\\nonsurvivor\\wm_boundary'
pial_output_dir = 'new_data\\nonsurvivor\\pial_boundary'

# Iterate over each pair of cortex and brain files
for cortex_file, brain_file in zip(cortex_files, brain_files):
    # Extract the identifier (e.g., "500" from "500_cortex.nii.gz")
    identifier = cortex_file.split('_')[0]

    print(f"Generating wm and pial boundaries for mouse {identifier}.")

    # Load data
    cortex_data = nib.load(os.path.join(cortex_dir, cortex_file)).get_fdata()
    whole_brain_data = nib.load(os.path.join(brain_dir, brain_file)).get_fdata()
    
    # Convert to boolean for operations
    cortex_bool = cortex_data.astype(bool)
    whole_brain_bool = whole_brain_data.astype(bool)

    # For the WM boundary:
    # Dilate the cortex mask
    wm_dilated = binary_dilation(cortex_bool, iterations=1)
    whole_brain_eroded = binary_erosion(whole_brain_bool, iterations = 2)

    # Restrict the dilation to the exterior of the cortex using the whole brain mask
    wm_boundary = wm_dilated & (~cortex_bool) & whole_brain_eroded

    # For the pial boundary:
    pial_dilated = binary_dilation(cortex_data, iterations=3)
    pial_boundary = pial_dilated & (~whole_brain_bool)

    # Convert boundaries to int and save
    wm_img = np.where(wm_boundary, 1, 0).astype(np.int16)
    pial_img = np.where(pial_boundary, 1, 0).astype(np.int16)

    # Save results to dictionaries
    wm_dict[identifier] = wm_img
    pial_dict[identifier] = pial_img

    output_filename = os.path.join(wm_output_dir, f"{identifier}_wm.nii.gz")
    cortex_img = nib.load(os.path.join(cortex_dir, cortex_file))
    output_image = nib.Nifti1Image(wm_img, cortex_img.affine)
    nib.save(output_image, output_filename)

    output_filename = os.path.join(pial_output_dir, f"{identifier}_pial.nii.gz")
    output_image = nib.Nifti1Image(pial_img, cortex_img.affine)
    nib.save(output_image, output_filename)
