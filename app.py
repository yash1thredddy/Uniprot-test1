import os
import shutil
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem import QED
from chembl_webresource_client.new_client import new_client
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Initialize ChEMBL clients
target = new_client.target
activity = new_client.activity
molecule = new_client.molecule

# Define global constants
ACTIVITY_TYPES = ["IC50", "EC50", "Ki", "Kd", "AC50", "GI50", "MIC"]

# Define reusable functions from your script
def extract_properties(smiles):
    if smiles == 'N/A':
        return np.nan, np.nan, np.nan
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan, np.nan, np.nan
        hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
        hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()
        return hbd, hba, heavy_atoms
    except:
        return np.nan, np.nan, np.nan

# Other helper functions like `get_target_info_from_uniprot` and `get_compounds_for_target`
# can stay the same or be refactored similarly.

def process_uniprot_target(uniprot_id):
    """Process a UniProt ID and return a summary and results"""
    folder_name = uniprot_id
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name)
    
    targets = get_target_info_from_uniprot(uniprot_id)
    if not targets:
        return None, f"No targets found for UniProt ID {uniprot_id}"
    
    # Collect data for Streamlit display
    target_summary = []
    all_results = []
    
    for target_info in targets:
        target_chembl_id = target_info['target_chembl_id']
        target_name = target_info.get('pref_name', 'Unknown')
        compounds, activities = get_compounds_for_target(target_chembl_id)
        target_summary.append({
            'ChEMBL Target ID': target_chembl_id,
            'Target Name': target_name,
            'Compound Count': len(compounds),
            'Activity Count': len(activities),
        })
        for chembl_id in compounds:
            results = fetch_and_calculate(chembl_id)
            all_results.extend(results)
    
    # Convert results to DataFrame for visualization
    if all_results:
        df_results = pd.DataFrame(all_results)
        return df_results, target_summary
    else:
        return None, target_summary

# Streamlit App
st.title("ChEMBL Analysis System")

# User input
uniprot_id = st.text_input("Enter a UniProt ID:", "")

if st.button("Analyze"):
    if uniprot_id.strip():
        with st.spinner("Processing..."):
            results, summary = process_uniprot_target(uniprot_id.strip())
        if results is not None:
            st.success("Processing complete!")
            
            # Display Target Summary
            st.subheader("Target Summary")
            st.write(pd.DataFrame(summary))
            
            # Display Results
            st.subheader("Compound Analysis Results")
            st.dataframe(results)
            
            # Downloadable CSV
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"{uniprot_id}_results.csv",
                mime='text/csv',
            )
        else:
            st.warning("No results generated.")
    else:
        st.error("Please enter a valid UniProt ID.")
