import os
import numpy as np
import pandas as pd
from ase import Atoms
from dscribe.descriptors import SOAP
import MDAnalysis as mda

#########################################################################################################
def calculate_descriptor_for_frame(frame_df, box):
    """
    Calculate SOAP descriptor for a given frame using Dscribe with dynamic box sizes.
    """
    symbols = frame_df['Element'].tolist()
    positions = frame_df[['x', 'y', 'z']].values

    # Create ASE Atoms object with variable cell size
    atoms = Atoms(symbols=symbols, positions=positions, cell=box, pbc=True)

    # Define the SOAP descriptor
    soap = SOAP(
        species=list(set(symbols)),
        periodic=True,
        r_cut=6.0,
        n_max=6,
        l_max=6,
        rbf="gto",
        sigma=1.0
    )

    # Calculate descriptor
    descriptor = soap.create(atoms)
    return descriptor

#########################################################################################################
def process_gromacs_trajectory(xtc_file, gro_file):
    """
    Process a GROMACS trajectory (XTC) and calculate SOAP descriptors for each frame.
    """
    u = mda.Universe(gro_file, xtc_file)
    descriptors = []
    frame_number = 0

    # Open file to store box dimensions
    cell_file = os.path.join(OUTPUT_FOLDER, "cell_dimensions.txt")
    with open(cell_file, "w") as f:
        f.write("Frame, Box Dimensions (Lx, Ly, Lz)\n")

        for ts in u.trajectory:
            box = ts.dimensions[:3]  # Extract box size (Lx, Ly, Lz)
            
            # Write cell dimensions to file
            f.write(f"{frame_number + 1}, {box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}\n")

            # Get atomic positions and elements
            atom_data = [{
                'Element': atom.name,
                'x': atom.position[0],
                'y': atom.position[1],
                'z': atom.position[2]
            } for atom in u.atoms]

            frame_df = pd.DataFrame(atom_data)
            descriptor = calculate_descriptor_for_frame(frame_df, box)
            descriptors.append({'frame': frame_number + 1, 'descriptor': descriptor})

            # Save descriptor to file
            descriptor_file = os.path.join(OUTPUT_FOLDER, f"descriptor_frame_{frame_number + 1}.npy")
            np.save(descriptor_file, descriptor)

            print(f"Frame {frame_number + 1}: Descriptor saved to {descriptor_file}, Box: {box}")
            frame_number += 1
    
    return frame_number

# Example Usage
OUTPUT_FOLDER = "./descriptors_output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

xtc_file = "./all.xtc"
gro_file = "./new.gro"

frame_count = process_gromacs_trajectory(xtc_file, gro_file)
print(f"Processed {frame_count} frames. Cell dimensions saved in cell_dimensions.txt")

