# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import country_converter as coco
import helpers
import numpy as np
import pandas as pd
import py7zr
import requests

# from helpers import configure_logging


# logger = logging.getLogger(__name__)


def download_urban_percent():
    """
    Downloads the United Nations "Total and urban population, annual" .7z File and extracts it as csv File.
    The above file was downloaded from the webpage https://unctadstat.unctad.org/datacentre/ as a .7z file. The dataset contains urban percent for most countries from 1950 and predictions until 2050.
    """
    url = "https://unctadstat-api.unctad.org/api/reportMetadata/US.PopTotal/bulkfile/355/en"

    # Make a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Extract the filename from the Content-Disposition header
        content_disposition = response.headers.get("Content-Disposition")
        if content_disposition:
            filename = content_disposition.split("filename=")[1].strip('"')
        else:
            filename = "downloaded_file.csv.7z"  # Provide a default filename if Content-Disposition header is not present

        # Write the content of the response to a file
        with open(filename, "wb") as f:
            f.write(response.content)

        print(f"Urban percent downloaded successfully as {filename}")

        # Extract the downloaded .7z file
        with py7zr.SevenZipFile(filename, "r") as archive:
            archive.extractall()

        print(f"Urban percent extracted successfully")

        # Read the extracted CSV file
        csv_filename = os.path.splitext(filename)[
            0
        ]  # Remove the .7z extension to get the CSV filename
        urban_percent_orig = pd.read_csv(csv_filename)

        print("Urban percent CSV file read successfully:")

        # Remove the downloaded .7z and .csv files
        os.remove(filename)
        os.remove(csv_filename)

    else:
        print(f"Failed to download file: Status code {response.status_code}")

    return urban_percent_orig


if __name__ == "__main__":
    if "snakemake" not in globals():
        from helpers import mock_snakemake, sets_path_to_root

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        snakemake = mock_snakemake("prepare_urban_percent")
        sets_path_to_root("pypsa-earth-sec")
    # configure_logging(snakemake)

    # run = snakemake.config.get("run", {})
    # RDIR = run["name"] + "/" if run.get("name") else ""
    # store_path_data = Path.joinpath(Path().cwd(), "data")
    # country_list = country_list_to_geofk(snakemake.config["countries"])'

    df = download_urban_percent().copy()

    # Select the columns that we need to keep
    df = df[
        [
            "Year",
            "Economy Label",
            "Absolute value in thousands",
            "Urban population as percentage of total population",
        ]
    ]

    # Keep only years above 2020
    df = df.loc[(df["Year"] >= 2020)]

    # Add ISO2 country code for each country
    cc = coco.CountryConverter()
    Economy_Label = pd.Series(df["Economy Label"])
    df["country"] = cc.pandas_convert(
        series=Economy_Label, to="ISO2", not_found="not found"
    )

    # Drop isos that were not found:
    df = df.loc[df["country"] != "not found"]

    df = df.set_index("country")

    # Save
    df.to_csv(snakemake.output[0], sep=",", encoding="utf-8", header="true")
