#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script imports data for levelling rod calibration. It constructs a dataframe
and an associated torch dataset that is suitable for ml-based training with either
pyro or calipy.
"""

import pandas as pd
import pickle 
import torch
import matplotlib.pyplot as plt


# Load the Excel file
filepath = "../data_stochastic_modelling/data_levelling_rod_calibration/dataset_levelling_rod_calibration.ods"
df = pd.read_excel(filepath, sheet_name="Tabelle1", skiprows=4)
main_df = df


# Set column headers to the 4th row (row 3 in Excel, already handled by skiprows)
headers = pd.read_excel(filepath, sheet_name="Tabelle1", nrows=5)
main_df.columns = headers.iloc[3]

# Drop completely empty columns and rows
main_df = main_df.dropna(axis=1, how='all').dropna(axis=0, how='all')

# Convert to list of dicts (each row = 1 calibration)
data_list_unformatted = main_df.to_dict(orient='records')

class DataList():
    def __init__(self, data_list):
        self.list = data_list
        
    def extract_subdicts(self, property_names_list):
        new_list = []
        for k in range(len(self.list)):
            subdict = {key: self.list[k][key] for key in property_names_list}
            new_list.append(subdict)
        return new_list
    
data_list = DataList(data_list_unformatted)
            
minimal_list = data_list.extract_subdicts(['Staff Type', 'Staff Serial Number', 'Overall Offset [µm]', 'Overall Scale [ppm]'])


   
    
# Support functions
# Remove entries containing nan
def clean_data_list(data_list):
    """
    Removes entries from the list of dicts that contain any NaN or None values.
    """
    return [
        entry for entry in data_list
        if all(not pd.isna(value) for value in entry.values())
    ]

# Compile to tensor
def data_list_to_tensor(data_list, keys):
    """
    Converts a list of dicts into a torch tensor by stacking selected keys.
    Assumes all keys are present and values are numeric.
    """
    tensor_data = [
        [entry[key] for key in keys]
        for entry in data_list
    ]
    return torch.tensor(tensor_data, dtype=torch.float32)
   

"""
    2. Preprocess data
"""


# Also harmonize staff type descriptions
def clean_data_list_advanced(data_list):
    """
    Cleans a list of dicts by:
      - Removing entries with any NaN/None values
      - Removing entries with staff_type == 'mittelwert' (case-insensitive)
      - Normalizing 'Staff Type' field: lowercase, no spaces
    """
    cleaned_list = []

    for entry in data_list:
        # Skip entries with any NaN or None values
        if any(pd.isna(v) for v in entry.values()):
            continue

        staff_type = entry.get("Staff Type")
        staff_id = str(entry.get("Staff Serial Number"))

        if isinstance(staff_type, str):
            norm_staff_type = staff_type.replace(" ", "").lower()
            norm_staff_id = staff_id.replace(" ", "").lower()

            # Skip if staff_type is 'mittelwert'
            if norm_staff_type == "mittelwert":
                continue

            # Update the normalized version
            entry["staff_type"] = norm_staff_type
            entry["staff_id"] = norm_staff_id

        cleaned_list.append(entry)

    return cleaned_list




# Assemble data

dlist = clean_data_list(data_list.list)
minimal_dlist = clean_data_list_advanced(minimal_list)
minimal_tensor = data_list_to_tensor(minimal_dlist, ['Overall Offset [µm]', 'Overall Scale [ppm]'])




# i) Aggregate names

# gwld, gwcl are super old; thye should not be considered canon
reduction_dict_classes = {
    'gpcl2': 'l2m',
    'gpcl3': 'l3m',
    'gpcl3(ref.)': 'l3m',
    'ld13': 't3m',
    'leicagpcl0.5mini': 'misc',
    'leicagpcl2': 'l2m',
    'leicagpcl3': 'l3m',
    'leicagwcl92': 'misc',
    'leicagwcl182': 'misc',
    'nedogpcl3': 'l3m',
    'trimbleld13': 't3m',
    'wildgpcl2': 'l2m',
    'wildgpcl3': 'l3m',
    'zeissgwld182': 'misc',
    'zeissgwld92': 'misc',
    'zeissld12': 't2m',
    'zeissld13': 't3m',
    'geozmidi': 'l2m',
    'geozmidi(0.9m)': 'misc',
    'geozmidi(1m)': 'misc',
    'geozmini': 'misc',
    'geozmini(0.2m)': 'misc'
}


# Build reduction dict for staff rod id's




# 1. Extract staff types
staff_types = [entry["Staff Type"] for entry in minimal_list]
staff_types_reduced = [reduction_dict_classes.get(name, 'misc') for name in staff_types]
unique_types = sorted(set(staff_types_reduced))
# type_to_color = {stype: plt.cm.tab10(i % 10) for i, stype in enumerate(unique_types)}

# 2. Extract serial numbers
serial_numbers = [entry['Staff Serial Number'] for entry in data_list.list]
# Get unique serials, assign numeric ID starting from 1
unique_serials = sorted(set([str(serial_number) for serial_number in serial_numbers]))
serial_to_id = {s: i + 1 for i, s in enumerate(unique_serials)}

# Step 3: Add 'staff_id' field to each entry
for entry in data_list.list:
    serial = entry['Staff Serial Number']
    entry['staff_id'] = serial_to_id[str(serial)]

def add_info_data_list(data_list):
    for entry in data_list:
        entry["staff_type_reduced"] = reduction_dict_classes.get(entry["staff_type"], 'misc')
    return data_list

minimal_list = add_info_data_list(minimal_dlist)

        
#Export data
with open("../data_stochastic_modelling/data_levelling_rod_calibration/data_list.pkl", "wb") as f:
    pickle.dump(dlist, f)

with open("../data_stochastic_modelling/data_levelling_rod_calibration/minimal_list.pkl", "wb") as f:
    pickle.dump(minimal_list, f)
    
torch.save(minimal_tensor, "../data_stochastic_modelling/data_levelling_rod_calibration/minimal_tensor.pt")
    