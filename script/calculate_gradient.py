import numpy as np
import nibabel as nib
import numba
import os

# Directory paths
laplace_dir = 'new_data\\nonsurvivor\\laplace\\'
#cortex_path = 'new_data\\resampled_542_cortex.nii.gz\\'
output_dir = 'new_data\\nonsurvivor\\gradient\\'

# Load the Laplace solution to obtain its affine transformation
#cortex_img = nib.load(cortex_path)

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

@numba.jit(nopython=True)
def compute_gradient(data):
    grad_x = np.zeros_like(data)
    grad_y = np.zeros_like(data)
    grad_z = np.zeros_like(data)

    x_max, y_max, z_max = data.shape

    for k in range(z_max - 1):
        for j in range(y_max - 1):
            for i in range(x_max - 1):
                # Calculate forward differences
                forward_diff_x = data[i + 1, j, k] - data[i, j, k]
                forward_diff_y = data[i, j + 1, k] - data[i, j, k]
                forward_diff_z = data[i, j, k + 1] - data[i, j, k]

                # Apply forward or backward differences based on condition
                grad_x[i, j, k] = forward_diff_x if forward_diff_x != 0 else data[i, j, k] - data[i - 1, j, k]
                grad_y[i, j, k] = forward_diff_y if forward_diff_y != 0 else data[i, j, k] - data[i, j - 1, k]
                grad_z[i, j, k] = forward_diff_z if forward_diff_z != 0 else data[i, j, k] - data[i, j, k - 1]

    # Handle the last elements with backward differences
    grad_x[x_max - 1, :, :] = data[x_max - 1, :, :] - data[x_max - 2, :, :]
    grad_y[:, y_max - 1, :] = data[:, y_max - 1, :] - data[:, y_max - 2, :]
    grad_z[:, :, z_max - 1] = data[:, :, z_max - 1] - data[:, :, z_max - 2]

    return grad_x, grad_y, grad_z

# Iterate over the files in the Laplace directory
for laplace_file in os.listdir(laplace_dir):
    if laplace_file.endswith('.nii.gz'):
        # Construct full file paths
        laplace_path = os.path.join(laplace_dir, laplace_file)
        output_path = os.path.join(output_dir, laplace_file.replace('laplace_img', 'gradient_map'))

        # Load data
        phi = nib.load(laplace_path).get_fdata()

        # Calculate the gradients
        grad_x, grad_y, grad_z = compute_gradient(phi)

        # Combine the gradients into a single gradient map.
        gradient_map = np.stack([grad_x, grad_y, grad_z], axis=0)

        # Calculate magnitude of the gradient vectors
        magnitudes = np.sqrt(gradient_map[0]**2 + gradient_map[1]**2 + gradient_map[2]**2)

        # To avoid division by zero, set zero magnitudes to a small value
        magnitudes[magnitudes == 0] = 1e-4

        # Normalize each component of the gradient
        normalized_gradient_map = gradient_map / magnitudes

        # Load the Laplace solution to obtain its affine transformation
        laplace_img = nib.load(laplace_path)

        # Save the normalized gradient map as a NIfTI file
        output_image = nib.Nifti1Image(normalized_gradient_map, laplace_img.affine)
        nib.save(output_image, output_path)
        print(f"Processed and saved gradient map for {laplace_file}")