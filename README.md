# Locality Sensitive Hashing-based Dataset Reduction for Deep Potential Training  

This repository contains all the Python scripts, machine-learned potential models, and input files required to reproduce the results of the paper:  

**"Locality Sensitive Hashing-based Dataset Reduction for Deep Potential Training"**  

## **Project Overview**  
In this work, we present a novel method based on locality-sensitive hashing designed to minimize the dataset size, thereby reducing the number of expensive quantum chemical calculations while preserving dataset diversity and accuracy. Our approach achieves dataset reductions of nearly an order of magnitude.  

## **Installation and Setup**  

### **1. Create the Conda Environment**  
To run the scripts, first create a Conda environment named **`lsh`** using the provided `environment.yml` file:  

```bash
conda env create -f environment.yml
conda activate lsh
 ```
## **Workflow**

### **Step 1: Generate SOAP Descriptors**

XYZ files (SOAP_xyz.py)
GROMACS XTC trajectory files (SOAP_gro.py)
Generates SOAP descriptors and saves them in descriptors_output/

For XYZ Files:
python3 Hashing/SOAP_xyz.py


For GROMACS XTC Files:
python3 Hashing/SOAP_gro.py

### **Step 2: Apply Locality Sensitive Hashing (LSH)**
After generating SOAP descriptors, run hashing.py to perform hashing:

python3 Hashing/hashing.py
#### **Parameters**
PCA Components: Number of dimensions to retain
Bin Width: Controls clustering sensitivity
This step Flattens and normalizes SOAP descriptors
	  Applies PCA for dimensionality reduction
	  Uses Locality-Sensitive Hashing (LSH) for dataset partitioning

Results are stored in hash_buckets_output/.

### **Step 3: Generate the Final Bucket List**
Finally, structure the hashed dataset using distil.py:

python Hashing/distil.py
This refines clusters and outputs a well-structured dataset.

