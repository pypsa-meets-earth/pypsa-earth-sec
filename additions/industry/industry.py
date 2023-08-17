#%%
import os
import requests
import pandas as pd
import country_converter as coco
from pathlib import Path
import openpyxl

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 70)


#%%
# change current directory
import os
import sys

module_path = os.path.abspath(os.path.join('../../../../')) # To import helpers
if module_path not in sys.path:
    sys.path.append(module_path+"/scripts")
    
from _helpers import sets_path_to_root, three_2_two_digits_country

sets_path_to_root("pypsa-earth")



#%%
# -------------
# CEMENT
# -------------
# The following excel file was downloaded from the following webpage https://www.cgfi.ac.uk/spatial-finance-initiative/geoasset-project/cement/ . The dataset contains 3117 cement plants globally.

fn = "https://www.cgfi.ac.uk/wp-content/uploads/2021/08/SFI-Global-Cement-Database-July-2021.xlsx" 
storage_options = {'User-Agent': 'Mozilla/5.0'}
cement_orig = pd.read_excel(fn, index_col=0, storage_options=storage_options, sheet_name='SFI_ALD_Cement_Database', header=0)


#%%
df = cement_orig.copy()
df = df[['country', 'iso3', 'latitude', 'longitude', 'status', 'plant_type', 'capacity', 'year', 'city']]
df = df.rename(columns={"country": "Country", "latitude": "y", "longitude": "x", "city": "location"})
df["unit"] = 'Kt/yr'
df["technology"] = 'Industry NMM Cement'
df["capacity"] = df["capacity"]*1000
# Keep only operating steel plants
df = df.loc[df['status'] == 'Operating']

# Create a column with iso2 country code
cc = coco.CountryConverter()
iso3 = pd.Series(df['iso3'])
df["country"] = cc.pandas_convert(series=iso3, to='ISO2')  

# Dropping the null capacities reduces the dataframe from 3000+  rows to 1672 rows
df = df.dropna(axis=0, subset=['capacity'])
df.to_csv(r'./documentation/notebooks/additions/industry/industry_cement.csv', sep=',', encoding='utf-8', header='true')


#%%
df = df.loc[df["country"].isin(["UA"])]
df = df.rename(columns={"status": "quality"})
df['Source'] = 'GLOBAL DATABASE OF CEMENT PRODUCTION ASSETS'

df_ukraine = df[['country', 'y', 'x', 'technology', 'capacity', 'unit', 'quality', 'location', 'Source']].set_index('country')

df_ukraine.to_csv(r'./documentation/notebooks/additions/industry/industry_cement_ukraine.csv', sep=',', encoding='utf-8', header='true')


#%%
# -------------
# STEEL
# -------------
# Global Steel Plant Tracker data set you requested from Global Energy Monitor from the link below:

# The following excel file was downloaded from the following webpage https://globalenergymonitor.org/wp-content/uploads/2023/03/Global-Steel-Plant-Tracker-2023-03.xlsx . The dataset contains 1433 Steel plants globally.

fn = "https://globalenergymonitor.org/wp-content/uploads/2023/03/Global-Steel-Plant-Tracker-2023-03.xlsx" 
storage_options = {'User-Agent': 'Mozilla/5.0'}
steel_orig = pd.read_excel(fn, index_col=0, storage_options=storage_options, sheet_name='Steel Plants', header=0)


#%%
df = steel_orig.copy()
df = df[['Plant name (English)', 'Country', 'Coordinates', 'Coordinate accuracy', 'Status', 'Start date', 'Plant age (years)', 'Nominal crude steel capacity (ttpa)',
	'Nominal BOF steel capacity (ttpa)','Nominal EAF steel capacity (ttpa)','Nominal OHF steel capacity (ttpa)','Nominal iron capacity (ttpa)','Nominal BF capacity (ttpa)',
    'Nominal DRI capacity (ttpa)','Ferronickel capacity (ttpa)','Sinter plant capacity (ttpa)','Coking plant capacity (ttpa)','Pelletizing plant capacity (ttpa)', 'Category steel product', 'Main production process']]

# Keep only operating steel plants
df = df.loc[df['Status'] == 'operating']

# Create a column with iso2 country code
cc = coco.CountryConverter()
Country = pd.Series(df['Country'])
df["country"] = cc.pandas_convert(series=Country, to='ISO2')  

# Split Coordeinates column into x and y columns
df[['y', 'x']] = df['Coordinates'].str.split(',', 1, expand=True)

# Drop Coordinates column as it contains a ',' and is not needed anymore
df = df.drop(columns='Coordinates', axis=1)

df.to_csv(r'./documentation/notebooks/additions/industry/industry_steel.csv', sep=',', encoding='utf-8', header='true')



#%%
# -------------
# OIL REFINERIES
# -------------
# The data were downloaded directly from arcgis server using a query found on this webpage: https://www.arcgis.com/home/item.html?id=a6979b6bccbf4e719de3f703ea799259&sublayer=0#data
# and https://www.arcgis.com/home/item.html?id=a917ac2766bc47e1877071f0201b6280

# The dataset contains 536 global Oil refineries.

import requests
import pandas as pd
import json
import pprint
import seaborn as sns
import matplotlib.pyplot as plt

base_url = "https://services.arcgis.com"
facts = "/jDGuO8tYggdCCnUJ/arcgis/rest/services/Global_Oil_Refinery_Complex_and_Daily_Capacity/FeatureServer/0/query?f=json&where=1%3D1&returnGeometry=false&spatialRel=esriSpatialRelIntersects&outFields=*&orderByFields=FID%20ASC&resultOffset=0&resultRecordCount=537&cacheHint=true&quantizationParameters=%7B%22mode%22%3A%22edit%22%7D"

first_response = requests.get(base_url+facts)
response_list=first_response.json()


#%%
data=[]
for response in response_list['features']:  
    data.append({
        "FID_": response['attributes'].get('FID_'),
        "Company": response['attributes'].get('Company'),
        "Name": response['attributes'].get('Name'),
        "City": response['attributes'].get('City'),
        "Facility": response['attributes'].get('Facility'),
        "Prov_State": response['attributes'].get('Prov_State'),
        "Country": response['attributes'].get('Country'),
        "Address": response['attributes'].get('Address'),
        "Zip": response['attributes'].get('Zip'),
        "County": response['attributes'].get('County'),
        "PADD": response['attributes'].get('PADD'),
        "Capacity": response['attributes'].get('Capacity'),
        "Longitude": response['attributes'].get('Longitude'),
        "Latitude": response['attributes'].get('Latitude'),
        "Markets": response['attributes'].get('Markets'),
        "CORPORATIO": response['attributes'].get('CORPORATIO')
    })

df = pd.DataFrame(data)

df.to_csv(r'./documentation/notebooks/additions/industry/industry_oil_refineries.csv', sep=',', encoding='utf-8', header='true')


#%%
# ---------------------------------------
# Coal and Ferrous mines and refineries
# ---------------------------------------

# The data were downloaded from Zenodo: https://zenodo.org/record/7369478#.ZEeszHZByUk
# and The arrical paper is in Nature: https://www.nature.com/articles/s41597-023-01965-y#Sec14

# This data set covers global extraction of coal and metal ores on an individual mine level. It covers
# 1171 individual mines in 80 different countries, reporting mine-level production for 80 different materials in the period 2000-2021.

# Furthermore, also data on mining coordinates, ownership, mineral reserves, mining waste, transportation of mining products, as well as mineral processing capacities (smelters and mineral refineries) and production is included. 


import country_converter as coco
import numpy as np

# Facililties with geo-locations:
fn = (os.getcwd() + "/documentation/notebooks/additions/industry/coal_and_ferrous/facilities.xlsx")
facilities = pd.read_excel(fn, index_col=0, header=0)

# Remove closed facililties
facilities = facilities.loc[facilities["production_end"].isnull()]

facilities = facilities.drop(['facility_other_names', 'sub_site_name','sub_site_other_names', 'production_start', 'production_end', 'activity_status', 'activity_status_year', 'surface_area_sq_km',
         'concession_area_sq_km', 'GID_0', 'GID_1', 'GID_2', 'GID_3', 'GID_4', 'source_id', 'comment'], axis=1)
facilities = facilities.reset_index()
facilities = facilities.replace(r'^\s*$', np.nan, regex=True)

# Add ISO2 country code for each country
facilities = facilities.rename(columns={"country": "Country"})
cc = coco.CountryConverter()
Country = pd.Series(facilities['Country'])
facilities["country"] = cc.pandas_convert(series=Country, to='ISO2') 



#%%
fn = (os.getcwd() + "/documentation/notebooks/additions/industry/coal_and_ferrous/capacity.csv")
capacity = pd.read_csv(fn)
capacity = capacity.drop('comment', axis=1)

# # Keep only considered year
# capacity = capacity.loc[capacity["year"].isin([year_considered])]

capacity = capacity[['facility_id','commodity', 'value_tpa']]
capacity = capacity.rename(columns={"value_tpa": "capacity_tpa"})


df1 = pd.merge(facilities, capacity, left_on='facility_id', right_on='facility_id', how='left')


#%%
# --------
# Coal
# --------

# coal:
fn = (os.getcwd() + "/documentation/notebooks/additions/industry/coal_and_ferrous/coal.csv")
coal = pd.read_csv(fn)

# Mean values for production over all years for each facility
df =coal.copy()
df = df[['facility_id', 'value_tonnes']].groupby(['facility_id']).max().reset_index()  # .mean()

# Drop duplicate facilities and keep only its material
coal = coal.sort_values(['facility_id'], ascending=[True]).drop_duplicates(['facility_id']).reset_index(drop=True)
coal = coal[['facility_id', 'material']]


coal = pd.merge(coal, df, left_on='facility_id', right_on='facility_id', how='left')

df2 = pd.merge(df1, coal, left_on='facility_id', right_on='facility_id', how='left')#.drop(columns = ['coal_facility_id'])


#%%
# --------
# Commodities
# --------

# commodities:
fn = (os.getcwd() + "/documentation/notebooks/additions/industry/coal_and_ferrous/commodities.csv")
commodities = pd.read_csv(fn)

# Mean values for production over all years for each facility
commodities = commodities.replace(r'^\s*$', np.nan, regex=True)
commodities = commodities[['facility_id', 'material', 'commodity', 'value_tonnes']].groupby(['facility_id', 'material', 'commodity'], dropna=False).max().reset_index()  # .mean()
commodities


#%%
df3 = pd.merge(df2, commodities, left_on='facility_id', right_on='facility_id', how='left')

df3['commodity'] = df3['commodity_x'].fillna(df3['commodity_y'])
df3['value_tonnes'] = df3['value_tonnes_x'].fillna(df3['value_tonnes_y'])
df3['material'] = df3['material_x'].fillna(df3['material_y'])
df3 = df3.drop(df3.filter(regex='_y$').columns, axis=1)
df3 = df3.drop(df3.filter(regex='_x$').columns, axis=1)


#%%
# --------
# Material_ids
# --------

# material_ids:
fn = (os.getcwd() + "/documentation/notebooks/additions/industry/coal_and_ferrous/material_ids.csv")
material_ids = pd.read_csv(fn)

# Mean values for production over all years for each facility
material_ids = material_ids.replace(r'^\s*$', np.nan, regex=True)
material_ids = material_ids[['material_id', 'material_name', 'material_category', 'material_category_2']]


df4 = pd.merge(df3, material_ids, left_on='material', right_on='material_id', how='left').drop(columns = ['material_id'])
df4


#%%
# --------
# Minerals
# --------

# minerals:
fn = (os.getcwd() + "/documentation/notebooks/additions/industry/coal_and_ferrous/minerals.csv")
minerals = pd.read_csv(fn)

# Mean values for production over all years for each facility
minerals = minerals.replace(r'^\s*$', np.nan, regex=True)
minerals = minerals[['facility_id', 'material', 'value_tonnes']].groupby(['facility_id', 'material' ], dropna=False).max().reset_index()  # .mean()

df5 = pd.merge(df4, minerals, left_on=['facility_id', 'material'], right_on=['facility_id', 'material'], how='left', suffixes=('','_material'))
df5



#%%
# --------
# Processing
# --------

# processing:
fn = (os.getcwd() + "/documentation/notebooks/additions/industry/coal_and_ferrous/processing.csv")
processing = pd.read_csv(fn)


processing = processing.replace(r'^\s*$', np.nan, regex=True)
# Mean values for production over all years for each facility
processing = processing[['facility_id', 'facility_type', 'input', 'input_value_tonnes', 'output', 'output_value_tonnes']].groupby(['facility_id', 'facility_type','input',  'output',], dropna=False).max().reset_index()  # .mean()

df6 = pd.merge(df5, processing, left_on=['facility_id', 'facility_type'], right_on=['facility_id', 'facility_type'], how='left', suffixes=('','_processing'))


#%%
df6.to_csv(r'./documentation/notebooks/additions/industry/industry_coal_and_ferrous.csv', sep=',', encoding='utf-8', header='true')


#%%

# # reserves:
# fn = (os.getcwd() + "/documentation/notebooks/additions/industry/coal_and_ferrous/reserves.csv")
# reserves = pd.read_csv(fn)
# reserves = reserves.add_prefix('reserves_')
# reserves

# # source_ids:
# fn = (os.getcwd() + "/documentation/notebooks/additions/industry/coal_and_ferrous/source_ids.csv")
# source_ids = pd.read_csv(fn)
# source_ids = source_ids.add_prefix('source_ids_')
# source_ids

# # transport:
# fn = (os.getcwd() + "/documentation/notebooks/additions/industry/coal_and_ferrous/transport.csv")
# transport = pd.read_csv(fn)
# transport = transport.add_prefix('transport_')
# transport

# # waste:
# fn = (os.getcwd() + "/documentation/notebooks/additions/industry/coal_and_ferrous/waste.csv")
# waste = pd.read_csv(fn)
# waste = waste.add_prefix('waste_')
# waste
