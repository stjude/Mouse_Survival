import torch
import nibabel as nib
import os

# Directory paths
# Path for saving the results
cortex_dir = 'new_data\\nonsurvivor\\cortex_mask'
wm_dir = 'new_data\\nonsurvivor\\wm_boundary\\'
pial_dir = 'new_data\\nonsurvivor\\pial_boundary\\'
output_dir = 'new_data\\nonsurvivor\\laplace\\'

def global_smoothing(tensor):
    i_plus = torch.roll(tensor, shifts=1, dims=0)
    i_minus = torch.roll(tensor, shifts=-1, dims=0)
    j_plus = torch.roll(tensor, shifts=1, dims=1)
    j_minus = torch.roll(tensor, shifts=-1, dims=1)
    k_plus = torch.roll(tensor, shifts=1, dims=2)
    k_minus = torch.roll(tensor, shifts=-1, dims=2)

    avg_neighbors = (i_plus + i_minus + j_plus + j_minus + k_plus + k_minus) / 6
    return avg_neighbors

def solve_laplace_torch(cortex_data, wm_boundary, pial_boundary):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    phi = torch.zeros_like(cortex_data, device=device)
    phi[wm_boundary] = 1
    phi[pial_boundary] = 0

    previous_phi = phi.clone()
    max_iterations = 2000
    threshold = 1e-6
    
    for iteration in range(max_iterations):
        phi_smooth = global_smoothing(phi)
        phi = torch.where(cortex_data.to(device) == 1, phi_smooth, phi)

        diff = torch.abs(phi - previous_phi)

        if iteration % 50 == 0:
            max_diff = torch.max(diff)
            print(f"Finished iteration number {iteration}, max difference = {max_diff}.")

        # Check for convergence
        if torch.max(diff) < threshold:
            print(f"Converged after {iteration} iterations.")
            break
        previous_phi = phi.clone()

    return phi.cpu()

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get the list of files
cortex_files = sorted(os.listdir(cortex_dir))
wm_files = sorted(os.listdir(wm_dir))
pial_files = sorted(os.listdir(pial_dir))

# Iterate over the files
for cortex_file in cortex_files:
    identifier = cortex_file.split('_')[0]  # Assuming the format is '500_cortex.nii.gz'
    print(f"Calculating laplace solution for mouse {identifier}.")

    # Corresponding file names
    wm_file = f"{identifier}_wm.nii.gz"
    pial_file = f"{identifier}_pial.nii.gz"
    output_file = f"{identifier}_laplace.nii.gz"
    
    # Construct full file paths
    cortex_path = os.path.join(cortex_dir, cortex_file)
    wm_path = os.path.join(wm_dir, wm_file)
    pial_path = os.path.join(pial_dir, pial_file)
    output_path = os.path.join(output_dir, output_file)
    
    # Load your data here
    cortex_img = nib.load(cortex_path)
    cortex_data = torch.tensor(cortex_img.get_fdata(), dtype=torch.float32)
    wm_boundary = torch.tensor(nib.load(wm_path).get_fdata().astype(bool), dtype=torch.float32)
    pial_boundary = torch.tensor(nib.load(pial_path).get_fdata().astype(bool), dtype=torch.float32)

    #Ensure the boundaries are boolean
    wm_boundary = nib.load(wm_path).get_fdata().astype(bool)
    pial_boundary = nib.load(pial_path).get_fdata().astype(bool)

    # Solve Laplace's equation
    phi = solve_laplace_torch(cortex_data, wm_boundary, pial_boundary)

    # Save the solution
    output_image = nib.Nifti1Image(phi.numpy(), cortex_img.affine)
    nib.save(output_image, output_path)
    print(f"Processed and saved Laplace solution for {identifier}")
