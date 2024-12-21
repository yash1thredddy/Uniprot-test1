import os
import pandas as pd
import numpy as np
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

# Functions
def extract_properties(smiles):
    """Extract HBD, HBA, and Heavy Atoms from SMILES."""
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

def get_target_info_from_uniprot(uniprot_id):
    """Get ChEMBL target information from UniProt ID."""
    try:
        targets = target.filter(target_components__accession=uniprot_id)
        return list(targets)
    except Exception as e:
        st.error(f"Error finding target for UniProt ID {uniprot_id}: {str(e)}")
        return []

def fetch_and_calculate(chembl_id):
    """Fetch molecule data and calculate properties for a given ChEMBL ID."""
    try:
        mol_data = molecule.get(chembl_id)
        if not mol_data:
            return []
        molecular_properties = mol_data.get('molecule_properties', {})
        molecular_weight = float(molecular_properties.get('full_mwt', np.nan))
        psa = float(molecular_properties.get('psa', np.nan))
        smiles = mol_data.get('molecule_structures', {}).get('canonical_smiles', 'N/A')
        molecule_name = mol_data.get('pref_name', 'Unknown Name')
        hbd, hba, heavy_atoms = extract_properties(smiles)
        npol = hbd + hba if hbd is not None and hba is not None else np.nan

        # Get bioactivities
        bioactivities = list(activity.filter(
            molecule_chembl_id=chembl_id,
            standard_type__in=ACTIVITY_TYPES
        ).only(
            'standard_value', 'standard_units', 'standard_type', 'target_chembl_id'
        ))

        results = []
        for act in bioactivities:
            if act['standard_value'] and act['standard_units'] == 'nM':
                value = float(act['standard_value'])
                activity_type = act['standard_type']
                pActivity = -np.log10(value * 1e-9)
                sei = pActivity / (psa / 100) if psa else np.nan
                bei = pActivity / (molecular_weight / 1000) if molecular_weight else np.nan
                results.append({
                    'ChEMBL ID': chembl_id,
                    'Molecule Name': molecule_name,
                    'SMILES': smiles,
                    'Molecular Weight': molecular_weight,
                    'TPSA': psa,
                    'Activity Type': activity_type,
                    'Activity (nM)': value,
                    'pActivity': pActivity,
                    'SEI': sei,
                    'BEI': bei,
                    'HBD': hbd,
                    'HBA': hba,
                    'Heavy Atoms': heavy_atoms,
                    'NPOL': npol
                })

        return results
    except Exception as e:
        st.error(f"Error processing {chembl_id}: {str(e)}")
        return []

def create_analysis_plots(df_results):
    """Display analysis plots in Streamlit."""
    st.subheader("Activity Distribution by Target and Type")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_results, x='Activity Type', y='pActivity')
    plt.xticks(rotation=45)
    plt.title("Activity Distribution")
    st.pyplot(plt)

    st.subheader("SEI vs BEI")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_results, x='SEI', y='BEI', hue='Activity Type', alpha=0.6)
    plt.title("SEI vs BEI")
    st.pyplot(plt)

def process_uniprot_target(uniprot_id):
    """Process a UniProt ID and analyze all associated compounds."""
    targets = get_target_info_from_uniprot(uniprot_id)
    if not targets:
        st.warning(f"No targets found for UniProt ID {uniprot_id}")
        return None, None

    st.write(f"Found {len(targets)} target(s) for UniProt ID {uniprot_id}")
    all_results = []

    for target_info in targets:
        target_chembl_id = target_info['target_chembl_id']
        st.write(f"Processing target: {target_chembl_id}")
        compounds = [act['molecule_chembl_id'] for act in activity.filter(
            target_chembl_id=target_chembl_id,
            standard_type__in=ACTIVITY_TYPES
        ).only('molecule_chembl_id')]

        for chembl_id in compounds:
            results = fetch_and_calculate(chembl_id)
            all_results.extend(results)

    if all_results:
        df_results = pd.DataFrame(all_results)
        return df_results, targets
    else:
        return None, targets

# Streamlit App
st.title("ChEMBL Analysis System")
st.markdown("Enter a UniProt ID to analyze associated compounds and targets.")

uniprot_id = st.text_input("Enter a UniProt ID:")

if st.button("Analyze"):
    if uniprot_id.strip():
        with st.spinner("Processing..."):
            df_results, targets = process_uniprot_target(uniprot_id.strip())

        if df_results is not None:
            st.success("Analysis Complete!")
            st.subheader("Compound Results")
            st.dataframe(df_results)

            # Downloadable CSV
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"{uniprot_id}_results.csv",
                mime='text/csv',
            )

            # Show Plots
            create_analysis_plots(df_results)
        else:
            st.warning("No results generated.")
    else:
        st.error("Please enter a valid UniProt ID.")
