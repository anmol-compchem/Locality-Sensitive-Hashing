bin_width = 0.0005
input_file_path = f'./hash_buckets_output/hash_buckets_flattened_{bin_width}.txt'
output_file_path = f'./hash_buckets_output/output_bins_{bin_width}.txt'


bin_to_frames = {}

with open(input_file_path, 'r') as file:
    #lines = file.readlines()[1:]
    for line in file:
        if line.strip():  # Skip empty lines
            # Extract frame number and bin index
            frame_number, bin_index = line.split(':')
            frame_number = int(frame_number.strip())  # Convert frame number to integer
            bin_index = int(bin_index.strip())  # Convert bin index to integer
            
            # Add frame number to the corresponding bin
            if bin_index not in bin_to_frames:
                bin_to_frames[bin_index] = []
            bin_to_frames[bin_index].append(frame_number)

# Calculate total frames and bins
total_frames = sum(len(frames) for frames in bin_to_frames.values())
total_bins = len(bin_to_frames)

# Write the sorted mapping to the output file
with open(output_file_path, 'w') as output_file:
    # Write summary information
    output_file.write(f"Total Frames: {total_frames}\n")
    output_file.write(f"Total Bins: {total_bins}\n\n")
    
    # Write bin-to-frames mapping
    for bin_index in sorted(bin_to_frames.keys()):
        frames = ', '.join(map(str, bin_to_frames[bin_index]))  # Convert frame numbers to a comma-separated string
        output_file.write(f"Bin {bin_index}: [{frames}]\n")

print(f"Output written to {output_file_path}")
