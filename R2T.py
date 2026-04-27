import os
import nibabel as nib
import numpy as np

def convert_t_maps_to_r_maps(input_dir, output_dir):
    """
    Convert T1_MAP and T2_MAP NIfTI files to R1_MAP and R2_MAP.
    Keep 0 as 0, convert others as 1/v.
    PD_map files are copied unchanged.
    """
    os.makedirs(output_dir, exist_ok=True)

    nii_files = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz')]
    for fname in nii_files:
        fpath = os.path.join(input_dir, fname)
        img = nib.load(fpath)
        img = nib.as_closest_canonical(img)
        data = img.get_fdata()

        # Initialize output
        out_data = None
        out_name = fname

        if "R1_MAP" in fname.upper():
            print(f"Converting R1 → T1 for: {fname}")
            out_data = np.zeros_like(data)
            nonzero_mask = data != 0
            out_data[nonzero_mask] = 1.0 / data[nonzero_mask]
            out_name = "qmap_t1.nii"

        elif "R2_MAP" in fname.upper():
            print(f"Converting R2 → T2 for: {fname}")
            out_data = np.zeros_like(data)
            nonzero_mask = data != 0
            out_data[nonzero_mask] = 1.0 / data[nonzero_mask]
            out_name = 'qmap_t2.nii'

        elif "PD_MAP" in fname.upper():
            print(f"Copying PD_map (no conversion): {fname}")
            out_data = data
            out_name = 'qmap_pd.nii'

        else:
            # Skip unrelated files
            continue

        # Save result
        out_path = os.path.join(output_dir, out_name)
        new_img = nib.Nifti1Image(out_data, img.affine, img.header)
        nib.save(new_img, out_path)


# Example usage
if __name__ == "__main__":
    input_dir = "/Users/amadeus/Downloads/SyMRI_processed_DL/Control_2"
    output_dir = "./data/Control_2_DL"
    convert_t_maps_to_r_maps(input_dir, output_dir)
