import os
import pandas as pd
from ase import Atoms
from dscribe.descriptors import SOAP
import numpy as np

#########################################################################################################
def calculate_descriptor_for_frame(frame_df):
    """
    Calculate SOAP descriptor for a given frame using Dscribe.
    Parameters:
        frame_df (DataFrame): DataFrame containing atomic coordinates and elements.
        metadata (dict): Metadata for the frame.
    Returns:
        descriptor (numpy.ndarray): Descriptor array for the frame.
    """
    # Create an ASE Atoms object from the frame data
    symbols = frame_df['Element'].tolist()
    positions = frame_df[['x', 'y', 'z']].values
    
    atoms = Atoms(symbols=symbols, positions=positions, cell=[15.0, 15.0, 20.0], pbc=True)
    
    soap = SOAP(
        species=list(set(symbols)),  # Unique atomic species
        periodic=True,
        r_cut=6.0,          # Reduced cutoff radius
        n_max=4,            # Reduce number of radial basis functions
        l_max=4,            # Reduce angular resolution
        rbf="gto",          # Radial basis function type
        sigma=1.0           # Gaussian smearing
    )
    
    # Calculate the descriptor
    descriptor = soap.create(atoms)
    
    return descriptor
#########################################################################################################
def process_xyz_with_descriptors(file_path):
    """
    Process an XYZ file and calculate descriptors for each frame.

    Parameters:
        file_path (str): Path to the .xyz file.

    Returns:
        descriptors (list): List of descriptors for each frame.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    i = 0
    frame_number = 0
    descriptors = []
    
    while i < len(lines):
        # Parse number of atoms and metadata
        num_atoms = int(lines[i].strip())
        metadata = lines[i + 1].strip()
        
        # Parse atomic data
        atom_data = []
        for j in range(i + 2, i + 2 + num_atoms):
            parts = lines[j].split()
            atom_data.append({
                'Element': parts[0],
                'x': float(parts[1]),
                'y': float(parts[2]),
                'z': float(parts[3])
            })
        
        # Create DataFrame for the frame
        frame_df = pd.DataFrame(atom_data)
        
        # Calculate descriptor
        descriptor = calculate_descriptor_for_frame(frame_df)
        descriptors.append({'frame': frame_number + 1, 'metadata': metadata,'descriptor': descriptor})
        
        # Save descriptor to the folder
        descriptor_file = os.path.join(OUTPUT_FOLDER, f"descriptor_frame_{frame_number + 1}.npy")
        np.save(descriptor_file, descriptor)
        
        # Print frame details
        #print(f"\n Frame {frame_number + 1}")
        #print(f"Metadata: {metadata}")
        #print(f"Descriptor saved to: {descriptor_file}")
        #print(f"Descriptor Shape: {descriptor.shape}")
        
        # Move to the next frame
        i += 2 + num_atoms
        frame_number += 1
    
    return frame_number


# Example Usage
# Create an output folder
OUTPUT_FOLDER = "./descriptors_output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
file_path = './all.xyz'
frame_number = process_xyz_with_descriptors(file_path)
print(frame_number)
