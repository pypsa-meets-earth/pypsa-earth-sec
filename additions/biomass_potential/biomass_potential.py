#%%
import os
import requests
import py7zr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import country_converter as coco
from pathlib import Path
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 70)

# change current directory
import os
import sys

module_path = os.path.abspath(os.path.join('../../../../pypsa-earth_old')) # To import helpers
if module_path not in sys.path:
    sys.path.append(module_path+"/scripts")
    
from _helpers import sets_path_to_root, three_2_two_digits_country

sets_path_to_root("Ukraine_old")


#%%
# The following xlsx file was downloaded from the webpage https://s2biom.wenr.wur.nl/web/guest/data-downloads.

fn = "https://s2biom.wenr.wur.nl/doc/data/data_UA.xlsx" 
storage_options = {'User-Agent': 'Mozilla/5.0'}
biomass_potential_ukraine = pd.read_excel(fn, index_col=0, storage_options=storage_options, sheet_name='dm_BASE_2030_kton', header=0)

biomass_potential_ukraine = biomass_potential_ukraine.reset_index().rename(columns={'index': 'ID_S2biom_DB_L3'})

biomass_potential_ukraine


#%%

regions = pd.read_excel(fn, index_col=0, storage_options=storage_options, sheet_name='readme_regions', header=0)
regions = regions[regions['ID_S2biom_DB_L3'].str.contains("UA")]

types = pd.read_excel(fn, index_col=0, storage_options=storage_options, sheet_name='readme', header=0, skiprows=range(27),nrows=50)
types = types.reset_index()


#%%
# Transpose the second DataFrame
biomass_potential_ukraine_transposed = biomass_potential_ukraine.transpose().reset_index() # .set_index('ID_S2biom_DB_L3')

# Set the first row as column headers
biomass_potential_ukraine_transposed.columns = biomass_potential_ukraine_transposed.iloc[0]
biomass_potential_ukraine_transposed = biomass_potential_ukraine_transposed[1:].reset_index(drop=True)

# Convert the 'ID_S2biom_DB_L3' from 'object' to 'int64'
biomass_potential_ukraine_transposed['ID_S2biom_DB_L3'] = biomass_potential_ukraine_transposed['ID_S2biom_DB_L3'].astype('int64')


#%%
# Merge the DataFrames on different column names
merged_df1 = pd.merge(biomass_potential_ukraine_transposed, types[['type_id', 'categorie']], left_on='ID_S2biom_DB_L3', right_on='type_id') # ['type_id', 'categorie', 'short_name'] --> if we want to have multiindex later
merged_df1


#%%
# Transpose the merged DataFrame
merged_df1_T = merged_df1.transpose().reset_index()

# Set the first row as column headers
merged_df1_T.columns = merged_df1_T.iloc[27]
merged_df1_T = merged_df1_T[1:].reset_index(drop=True)

# Drop rows by index number
merged_df1_T = merged_df1_T.drop([25, 26])

# Merge the DataFrames on different column names
merged_df = pd.merge(merged_df1_T, regions[['ID_S2biom_DB_L3', 'name']], left_on='categorie', right_on='ID_S2biom_DB_L3')
merged_df


#%%
# Merge the DataFrames on different column names
df = pd.merge(biomass_potential_ukraine, regions[['ID_S2biom_DB_L3', 'name']], left_on='ID_S2biom_DB_L3', right_on='ID_S2biom_DB_L3')
df

#%%
solid_biomass_potential = df [['name','1111', '1112', '1113', '1114', '1211', '1212', '1213', '1214', '1221', '1222']].groupby(['name'])['1111', '1112', '1113', '1114', '1211', '1212', '1213', '1214', '1221', '1222'].sum().sum(axis=1)
solid_biomass_potential

#%%
solid_biomass_potential = df[['name','1111', '1112', '1113', '1114', '1211', '1212', '1213', '1214', '1221', '1222']].set_index('name')

# Calculate the sum of all columns
solid_biomass_potential['solid_biomass_pot'] = solid_biomass_potential.sum(axis=1)

# Drop all other columns except the sum column
solid_biomass_potential = solid_biomass_potential[['solid_biomass_pot']]

solid_biomass_potential

#%%
solid_biomass_potential.sum()


#%%
biogas_potential = df[['name','2112', '2113', '2114', '2115', '2116', '5111', '5112']].set_index('name')

# Calculate the sum of all columns
biogas_potential['biogas_potential'] = biogas_potential.sum(axis=1)

# Drop all other columns except the sum column
biogas_potential = biogas_potential[['biogas_potential']]

biogas_potential

#%%
biogas_potential.sum()


#%%
not_included_potential = df.set_index('name').sum(axis=1) - biogas_potential['biogas_potential'] - solid_biomass_potential['solid_biomass_pot']
not_included_potential


#%%
not_included_potential.sum()
