import numpy as np
import nibabel as nib
import os

# Directory paths
gradient_dir = 'new_data\\survivor\\gradient\\'
laplace_dir = 'new_data\\survivor\\laplace\\'
pial_dir = 'new_data\\survivor\\pial_boundary\\'
cortex_dir = 'new_data\\survivor\\cortex_mask\\'
output_dir = 'new_data\\survivor\\thickness_maps\\'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load filenames from each folder
gradient_files = sorted([f for f in os.listdir(gradient_dir) if f.endswith('.nii.gz')])
laplace_files = sorted([f for f in os.listdir(laplace_dir) if f.endswith('.nii.gz')])
pial_files = sorted([f for f in os.listdir(pial_dir) if f.endswith('.nii.gz')])
cortex_files = sorted([f for f in os.listdir(cortex_dir) if f.endswith('.nii.gz')])

def compute_thickness_map(phi, gradient_map, pial_boundary):
    # Initialize an empty thickness map.
    thickness_map = np.zeros_like(phi)

    # Initialize an empty paths dictionary.
    paths = {}

    # Define a threshold for identifying the boundaries.
    WM_THRESHOLD = 1

    pialCount = 0
    wm_count = 0

    # Iterate through each voxel in the image.
    for k in range(phi.shape[2]):
        for i in range(phi.shape[0]):
            for j in range(phi.shape[1]):
                # Check if the voxel is on the pial boundary.
                if pial_boundary[i, j, k] == 1:
                    pialCount += 1
                    x, y, z = i, j, k  # Initialize coordinates for tracing the path.
                    thickness = 0  # Initialize thickness counter.
                    current_path = []  # Create a list to store the path.

                    # Follow the gradient path.
                    while phi[int(x), int(y), int(z)] < WM_THRESHOLD and thickness < 200:
                        grad_x = gradient_map[0, int(x), int(y), int(z)]
                        grad_y = gradient_map[1, int(x), int(y), int(z)]
                        grad_z = gradient_map[2, int(x), int(y), int(z)]

                        current_path.append((int(x), int(y), int(z)))  # Append current voxel to the path.
                        
                        # Move along the gradient
                        x += grad_x * 0.1
                        y += grad_y * 0.1
                        z += grad_z * 0.1
                        
                        thickness += 1  # Increment the thickness counter.

                    thickness_map[i, j, k] = thickness  # Store the calculated thickness.
                    paths[(i, j, k)] = current_path  # Store the path in the paths directory.
                    
                    if WM_THRESHOLD == 1:
                        wm_count += 1

    print(pialCount, wm_count)
    return thickness_map, paths

# Iterate over each set of files
for gradient_file, laplace_file, pial_file, cortex_file in zip(gradient_files, laplace_files, pial_files, cortex_files):
    identifier = gradient_file.split('_')[0]  # Assuming files are named like 'identifier_gradient_map.nii.gz'

    print(f"Processing thickness map for {identifier}")

    # Construct the full paths for each file
    gradient_path = os.path.join(gradient_dir, gradient_file)
    laplace_path = os.path.join(laplace_dir, laplace_file)
    pial_path = os.path.join(pial_dir, pial_file)
    cortex_path = os.path.join(cortex_dir, cortex_file)
    output_path = os.path.join(output_dir, f"{identifier}_thickness_map.nii.gz")

    # Load the data for the current mouse
    gradient_map = nib.load(gradient_path).get_fdata()
    phi = nib.load(laplace_path).get_fdata()
    pial_boundary = nib.load(pial_path).get_fdata()

    # Compute the thickness map using the loaded data
    thickness_map, voxel_paths = compute_thickness_map(phi, gradient_map, pial_boundary)

    # Set thicknesses that failed to find WM as NaN
    thickness_map[thickness_map == 200] = np.nan

    # Load the cortex mask to obtain its affine transformation
    cortex_img = nib.load(cortex_path)

    # Save the thickness map as a NIfTI file
    output_image = nib.Nifti1Image(thickness_map, cortex_img.affine)
    nib.save(output_image, output_path)
    print(f"Saved thickness map for {identifier} at {output_path}")
