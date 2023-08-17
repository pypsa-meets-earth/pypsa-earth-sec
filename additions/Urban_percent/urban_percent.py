# -*- coding: utf-8 -*-
# %%
import os

import country_converter as coco
import pandas as pd
import py7zr
import requests

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 70)

import sys

module_path = os.path.abspath(
    os.path.join("../../../../pypsa-earth_old")
)  # To import helpers
if module_path not in sys.path:
    sys.path.append(module_path + "/scripts")

from _helpers import sets_path_to_root, three_2_two_digits_country

sets_path_to_root("Ukraine")


# %%
# The following csv file was downloaded from the same webpage https://unctadstat.unctad.org/EN/BulkDownload.html as a .7z file.

# Download zipfile and unzip it
URL = "https://unctadstat.unctad.org/7zip/US_PopTotal.csv.7z"
filename = os.path.basename(URL)

response = requests.get(URL, stream=True)

if response.status_code == 200:
    with open(filename, "wb") as out:
        out.write(response.content)
    with py7zr.SevenZipFile(filename, "r") as archive:
        archive.extractall(f"./documentation/notebooks/additions/Urban_percent")
else:
    print("Request failed: %d" % response.status_code)


# %%
# Read downloaded csv
urban_percent_orig = pd.read_csv(
    r"./documentation/notebooks/additions/Urban_percent/US_PopTotal_ST202210141520_v2.csv",
    index_col=0,
)


# %%
df = urban_percent_orig.copy()
df = df[
    [
        "Year",
        "Economy Label",
        "Absolute value in thousands",
        "Urban population as percentage of total population",
    ]
]

df = df.loc[(df["Year"] == 2030) | (df["Year"] == 2050)]  #  & (df['Year'] == 2050)

cc = coco.CountryConverter()

Economy_Label = pd.Series(df["Economy Label"])

df["country"] = cc.pandas_convert(
    series=Economy_Label, to="ISO2", not_found="not found"
)

df = df.loc[df["country"] != "not found"]


# %%
# For Ukraine
df[df["country"] == "UA"]


# %%
# Save
df.to_csv(
    r"./documentation/notebooks/additions/Urban_percent/urban_percent.csv",
    sep=",",
    encoding="utf-8",
    header="true",
)
