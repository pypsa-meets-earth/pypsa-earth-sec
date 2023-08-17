# -*- coding: utf-8 -*-
# %%

import os
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import country_converter as coco
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py7zr
import requests

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 70)


# change current directory
import os
import sys

module_path = os.path.abspath(
    os.path.join("../../../../pypsa-earth_old")
)  # To import helpers
if module_path not in sys.path:
    sys.path.append(module_path + "/scripts")

from _helpers import sets_path_to_root, three_2_two_digits_country

sets_path_to_root("Ukraine")


# %%

# The following json file was downloaded from the following webpage https://apps.who.int/gho/data/node.main.

import json
import pprint

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns

base_url = "https://apps.who.int/gho/athena/data/GHO/RS_194.json"
facts = "?profile=simple&filter=COUNTRY:*&ead="

first_response = requests.get(base_url + facts)
response_list = first_response.json()

# %%
data = []
for response in response_list["fact"]:
    data.append(
        {
            "Country": response["dim"].get("COUNTRY"),
            # "Indicator": response['dim'].get('GHO'),
            # "Data Source": response['dim'].get('DATASOURCE'),
            # "Residence area type": response['dim'].get('RESIDENCEAREATYPE'),
            # "WHO region": response['dim'].get('REGION'),
            # "Year": response['dim'].get('YEAR'),
            # "PUBLISH STATES": response['dim'].get('PUBLISHSTATE'),
            "number cars": response["Value"],
        }
    )
df = pd.DataFrame(data)


# %%

# Add ISO2 country code for each country

cc = coco.CountryConverter()

Country = pd.Series(df["Country"])

df["country"] = cc.pandas_convert(series=Country, to="ISO2", not_found="not found")

# # Remove spaces, Replace empty values with NaN
df["number cars"] = df["number cars"].str.replace(" ", "").replace("", np.nan)

# Drop rows with NaN values in 'Column1'
df = df.dropna(subset=["number cars"])

# convert the 'number cars' to integer
df["number cars"] = df["number cars"].astype(int)


# %%
# The dataset is downloaded from the following link: https://data.worldbank.org/indicator/EN.CO2.TRAN.ZS?view=map
# It is up until the year 2014. # TODO: Maybe search for more recent years.


url = "https://api.worldbank.org/v2/en/indicator/EN.CO2.TRAN.ZS?downloadformat=excel"

response = requests.get(url)

if response.status_code == 200:
    with open(
        "/nfs/home/edd32710/projects/HyPAT/Ukraine/documentation/notebooks/additions/transport_data/CO2_emissions.xls",
        "wb",
    ) as f:
        f.write(response.content)
        print("File downloaded successfully.")
else:
    print("Failed to download the file.")


# Read the 'Data' sheet from the downloaded Excel file
CO2_emissions = pd.read_excel(
    "/nfs/home/edd32710/projects/HyPAT/Ukraine/documentation/notebooks/additions/transport_data/CO2_emissions.xls",
    sheet_name="Data",
    skiprows=[0, 1, 2],
)
CO2_emissions


# %%
CO2_emissions = CO2_emissions[
    ["Country Name", "Country Code", "Indicator Name", "2014"]
]


# %%
CO2_emissions = CO2_emissions.copy()

CO2_emissions["average fuel efficiency"] = (100 - CO2_emissions["2014"]) / 100

# Add ISO2 country code for each country

CO2_emissions = CO2_emissions.rename(columns={"Country Name": "Country"})

cc = coco.CountryConverter()

Country = pd.Series(CO2_emissions["Country"])

CO2_emissions["country"] = cc.pandas_convert(
    series=Country, to="ISO2", not_found="not found"
)

# Drop region names that have no ISO2:
CO2_emissions = CO2_emissions[CO2_emissions.country != "not found"]


# %%
# Join the DataFrames by the 'country' column
merged_df = pd.merge(df, CO2_emissions, on="country")

merged_df = merged_df[["country", "number cars", "average fuel efficiency"]]


# %%
# Save the DataFrame to a CSV file
merged_df.to_csv(
    "/nfs/home/edd32710/projects/HyPAT/Ukraine/documentation/notebooks/additions/transport_data/transport_data.csv",
    index=False,
)
