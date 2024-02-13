import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics.pairwise import pairwise_distances
#%%
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="7"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#%%
compound_df = pd.read_csv("/home/sabir/DDI_project/DDI_end/Dataprocess_file.csv")
print(compound_df.head())
#%%
# Create fingerprints from SMILES and store in array
def fingerprints_to_numpy(fingerprints, n_bits=2048):
    num_fps = len(fingerprints)
    np_fps = np.zeros((num_fps, n_bits), dtype=np.int32)

    for i, fp in enumerate(fingerprints):
        on_bits = list(fp.GetOnBits())
        np_fps[i, on_bits] = 1

    return np_fps
def drug_compound(drug_name):
    compounds = []
    for smiles in compound_df[drug_name]:
        compounds.append((Chem.MolFromSmiles(smiles)))
    # Create fingerprints for all smiles of drug Id's
    rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5)
    fingerprints = [rdkit_gen.GetFingerprint(mol) for mol in compounds]
    numpy_fingerprints = fingerprints_to_numpy(fingerprints)

    return numpy_fingerprints
#%%
fingerprint_drug1 = drug_compound('Drug1')
fingerprint_drug2 = drug_compound('Drug2')
fingerprint_drug1.shape
fingerprint_drug2.shape
#%%
np.save("Drug1_fn.npy",fingerprint_drug1)
np.save("Drug2_fn.npy",fingerprint_drug2)
#%%
combine_fingerprint= np.concatenate((fingerprint_drug1,fingerprint_drug2),axis=1)
combine_fingerprint.shape
#%%
compound_df.columns.values
#%%
label = compound_df['Y']
np.save('labels.npy', label)
#%%
np.save("features.npy",combine_fingerprint)
#%%
# Load Morgan fingerprints for drug1 and drug2
data_directory = "/home/sabir/"
drug1 = np.load(os.path.join(data_directory, "Drug1_fn.npy"), allow_pickle=True).astype(bool)
drug2 = np.load(os.path.join(data_directory, "Drug2_fn.npy"), allow_pickle=True).astype(bool)
#%%
# Function to calculate Tanimoto similarity and accumulate histogram data
def calculate_similarity_and_accumulate_data(drug1, drug2, bins=50):
    hist_data = np.zeros(bins)
    bin_edges = np.linspace(0, 1, bins + 1)
    
    for i in range(len(drug1)):
        # Compute pairwise similarity for each fingerprint in drug1 against all in drug2
        batch_similarity = 1 - pairwise_distances(drug1[i:i+1], drug2, metric='jaccard')
        # Accumulate histogram data
        batch_hist, _ = np.histogram(batch_similarity, bins=bin_edges)
        hist_data += batch_hist
    
    return hist_data, bin_edges
#%%
# Calculate similarity and accumulate histogram data
bins = 50
hist_data, bin_edges = calculate_similarity_and_accumulate_data(drug1, drug2, bins)
#%%
# Plotting
plt.figure(figsize=(10, 6))
plt.bar(bin_edges[:-1], hist_data, width=1/bins, align='edge', color='purple', edgecolor='black')
plt.title('Structural Similarity Profile (SSP) between Drug A and Drug B')
plt.xlabel('Tanimoto Similarity Score')
plt.ylabel('Frequency')
plt.show()
#%%
data_directory = "/home/sabir/DDI_project/DDI_end/"
# Load features and labels
features = np.load(os.path.join(data_directory, "features.npy"), allow_pickle=True)
labels = np.load(os.path.join(data_directory, "labels.npy"))
#%%
# Count the occurrences of each unique label
unique, counts = np.unique(labels, return_counts=True)
# Create a dictionary from labels to counts for easier interpretation
label_counts = dict(zip(unique, counts))
# Print the number of samples for every class
for label, count in label_counts.items():
    print(f"Class {label}: {count} samples")
#%%
# Define a list of colors
colors = ['orange','blue', 'purple', 'red', 'green','pink',
          'lightgreen', 'black', 'skyblue', 'yellow', 'grey', 'brown']
#%%
sorted_indices = np.argsort(-counts)  # Negative for descending order
sorted_unique = unique[sorted_indices]
sorted_counts = counts[sorted_indices]
#%%
# Repeat the color list to match the number of unique labels
color_list = np.resize(colors, len(sorted_unique))
# Plotting the sorted bar chart
plt.figure(figsize=(24, 14))
bars = plt.bar(range(len(sorted_unique)), sorted_counts, color=color_list)

plt.title('Distribution of Drug-Pairs across DDI Classes', fontsize=18)
plt.xlabel('DDI Class Number', fontsize=16)
plt.ylabel('Number of Drug-Pair Samples', fontsize=16)
plt.xticks(range(len(sorted_unique)), sorted_unique, rotation=90, fontsize=8)

plt.ylim([-50, max(sorted_counts)])

# Adding counts above bars
for bar, count in zip(bars, sorted_counts):
    height = bar.get_height()
    plt.annotate(f'{count}',                    # Text to annotate
                 xy=(bar.get_x() + bar.get_width() / 2, height),  # Position (x,y)
                 xytext=(0, 3),                # Text offset
                 textcoords="offset points",   # How to interpret the text position
                 ha='center', va='bottom',     # Alignment
                 fontsize=10,rotation=60)                   # Text size

plt.tight_layout()
plt.show()
#%%
