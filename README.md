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
Workflow
## Algorithm: Processing of SOAP Descriptors with Dimensionality Reduction and LSH

### **Inputs**
- **SOAP matrix** \( S \in \mathbb{R}^{N \times d} \) for each configuration in a dataset of \( X \) configurations
- **Target reduced dimension** \( D \)
- **Bin width** \( w \) for hash computation
- **Number of hash functions** \( H \)
- **Random bias vector** \( b \sim U(-w, w) \)

### **Outputs**
- Buckets containing similar configuration vectors.

---

### **Step 1: Flattening SOAP Descriptors**
1. Compute SOAP matrix \( S \in \mathbb{R}^{N \times d} \) for each configuration.
2. Flatten \( S \) into a configuration vector \( c \in \mathbb{R}^{(N \times d)} \).
3. Stack all configuration vectors into a matrix \( C \in \mathbb{R}^{X \times (N \times d)} \).
4. Normalize \( C \) row-wise.

---

### **Step 2: Dimensionality Reduction (PCA)**
1. Apply PCA transformation to reduce \( C \) to \( C' \in \mathbb{R}^{X \times D} \).
2. Normalize \( C' \) row-wise for stable hashing.

---

### **Step 3: Locality-Sensitive Hashing (LSH)**
#### **3.1 Generate Hash Functions**
- Generate a random projection matrix \( W_H \in \mathbb{R}^{D \times N_H} \), where:
  - \( W_H(i,j) \sim \mathcal{N}(0,1) \).

#### **3.2 Compute Hash Values**
- For each row vector \( c'_i \in C' \):
  1. Project the vector:  
     \( h_i = c'_i W^T_H \).
  2. Apply binning with bias \( b \):  
     \( H_i = \lfloor (h_i + b) / w \rfloor \).

#### **3.3 Assign Buckets**
- For each row vector \( c'_i \):
  1. Assign it to a bucket using:  
     \( n_i = \max(H_i) \).
  2. Store \( c'_i \) in the bucket labeled by \( n_i \).

---

### **Step 4: Output Buckets**
- Store all buckets in a dictionary where:
  - **Keys** = hash values.
  - **Values** = corresponding configuration vectors.
- **Return:** Buckets containing similar configurations.

---

This version will **render correctly on GitHub** without LaTeX issues. 🚀 Let me know if you want any modifications!


Step 1: Generate SOAP Descriptors

XYZ files (SOAP_xyz.py)
GROMACS XTC trajectory files (SOAP_gro.py)
Generates SOAP descriptors and saves them in descriptors_output/

For XYZ Files:
python3 Hashing/SOAP_xyz.py


For GROMACS XTC Files:
python3 Hashing/SOAP_gro.py

Step 2: Apply Locality Sensitive Hashing (LSH)
After generating SOAP descriptors, run hashing.py to perform hashing:

python3 Hashing/hashing.py
Parameters:
PCA Components: Number of dimensions to retain
Bin Width: Controls clustering sensitivity
This step Flattens and normalizes SOAP descriptors
	  Applies PCA for dimensionality reduction
	  Uses Locality-Sensitive Hashing (LSH) for dataset partitioning

Results are stored in hash_buckets_output/.

Step 3: Generate the Final Bucket List
Finally, structure the hashed dataset using distil.py:

python Hashing/distil.py
This refines clusters and outputs a well-structured dataset.

