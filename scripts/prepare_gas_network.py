# -*- coding: utf-8 -*-
"""
Prepare gas network.
"""

import logging

logger = logging.getLogger(__name__)

import os
import geopandas as gpd
import pandas as pd
from packaging.version import Version, parse
from pypsa.geo import haversine_pts
from shapely import wkt
from pyproj import CRS
from shapely.ops import unary_union
from shapely.geometry import LineString, MultiLineString
from shapely.geometry import Point
from matplotlib.lines import Line2D
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import zipfile
from pathlib import Path
from helpers import (progress_retrieve, two_2_three_digits_country)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from helpers import mock_snakemake, sets_path_to_root

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        snakemake = mock_snakemake(
            "prepare_gas_network",
            simpl="",
            clusters="28",
        )
        sets_path_to_root("pypsa-earth-sec")
        rootpath = ".."
    else:
        rootpath = "."
    # configure_logging(snakemake)

    # run = snakemake.config.get("run", {})
    # RDIR = run["name"] + "/" if run.get("name") else ""
    # store_path_data = Path.joinpath(Path().cwd(), "data")
    # country_list = country_list_to_geofk(snakemake.config["countries"])'


def download_IGGIELGN_gas_network():
    """
    Downloads a global dataset for gas networks as .xlsx.
    The following xlsx file was downloaded from the webpage https://globalenergymonitor.org/projects/global-gas-infrastructure-tracker/
    The dataset contains 3144 pipelines.
    """

    url = "https://zenodo.org/record/4767098/files/IGGIELGN.zip"

    # Save locations
    zip_fn = Path(f"{rootpath}/IGGIELGN.zip")
    to_fn = Path(f"{rootpath}/data/gas_network/scigrid-gas")

    # zip_fn = Path("/nfs/home/edd32710/projects/HyPAT/PES/pypsa-earth-sec/IGGIELGN.zip")
    # to_fn = Path("/nfs/home/edd32710/projects/HyPAT/PES/pypsa-earth-sec/data/gas_network/scigrid-gas")

    logger.info(f"Downloading databundle from '{url}'.")
    progress_retrieve(url, zip_fn)

    logger.info(f"Extracting databundle.")
    zipfile.ZipFile(zip_fn).extractall(to_fn)

    zip_fn.unlink()

    logger.info(f"Gas infrastructure data available in '{to_fn}'.")


def download_GGIT_gas_network():
    """
    Downloads a global dataset for gas networks as .xlsx.
    The following xlsx file was downloaded from the webpage https://globalenergymonitor.org/projects/global-gas-infrastructure-tracker/
    The dataset contains 3144 pipelines.
    """
    fn = "https://globalenergymonitor.org/wp-content/uploads/2022/12/GEM-GGIT-Gas-Pipelines-December-2022.xlsx" 
    storage_options = {"User-Agent": "Mozilla/5.0"}
    GGIT_gas_pipeline = pd.read_excel(
        fn, index_col=0, storage_options=storage_options, sheet_name='Gas Pipelines 2022-12-16', header=0
    )

    return (GGIT_gas_pipeline)



def diameter_to_capacity(pipe_diameter_mm):
    """
    Calculate pipe capacity in MW based on diameter in mm.

    20 inch (500 mm)  50 bar -> 1.5   GW CH4 pipe capacity (LHV) 24 inch
    (600 mm)  50 bar -> 5     GW CH4 pipe capacity (LHV) 36 inch (900
    mm)  50 bar -> 11.25 GW CH4 pipe capacity (LHV) 48 inch (1200 mm) 80
    bar -> 21.7  GW CH4 pipe capacity (LHV)

    Based on p.15 of
    https://gasforclimate2050.eu/wp-content/uploads/2020/07/2020_European-Hydrogen-Backbone_Report.pdf
    """
    # slopes definitions
    m0 = (1500 - 0) / (500 - 0)
    m1 = (5000 - 1500) / (600 - 500)
    m2 = (11250 - 5000) / (900 - 600)
    m3 = (21700 - 11250) / (1200 - 900)

    # intercept
    a0 = 0
    a1 = -16000
    a2 = -7500
    a3 = -20100

    if pipe_diameter_mm < 500:
        return a0 + m0 * pipe_diameter_mm
    elif pipe_diameter_mm < 600:
        return a1 + m1 * pipe_diameter_mm
    elif pipe_diameter_mm < 900:
        return a2 + m2 * pipe_diameter_mm
    else:
        return a3 + m3 * pipe_diameter_mm


def inch_to_mm(len_inch):
    return len_inch/0.0393701


def bcm_to_MW(cap_bcm):
    return (cap_bcm*9769444.44/8760)  


def correct_Diameter_col(value):
    value = str(value)
    # Check if the value contains a comma
    if ',' in value:
        # Split the value by comma and convert each part to a float
        diameter_values = [float(val) for val in value.split(',')]
        # Calculate the mean of the values
        return sum(diameter_values) / len(diameter_values)
    elif "/" in value:
        # Split the value by slash and convert each part to a float
        diameter_values = [float(val) for val in value.split("/")]
        # Calculate the mean of the values
        return sum(diameter_values) / len(diameter_values)
    elif "-" in value:
        # Split the value by slash and convert each part to a float
        diameter_values = [float(val) for val in value.split("-")]
        # Calculate the mean of the values
        return sum(diameter_values) / len(diameter_values)
    else:
        # Return the original value for rows without a comma or slash
        return float(value)



def prepare_GGIT_data(GGIT_gas_pipeline):
    df = GGIT_gas_pipeline.copy().reset_index()

    # Drop rows containing "--" in the 'WKTFormat' column
    df = df[df['WKTFormat'] != '--']

    # Keep pipelines that are as below
    df = df[(df['Status'] == 'Construction') | (df['Status'] == 'Operating') | (df['Status'] == 'Idle') | (df['Status'] == 'Shelved') | (df['Status'] == 'Mothballed')]

    # Convert the WKT column to a GeoDataFrame
    df = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df['WKTFormat']))

    # Set the CRS to EPSG:4326
    df.crs = CRS.from_epsg(4326)

    # Convert CRS to EPSG:3857 so we can measure distances
    df = df.to_crs(epsg=3857)

    # Convert and correct diameter column to be in mm
    df.loc[df['DiameterUnits'] == 'mm', 'diameter_mm'] = df.loc[df['DiameterUnits'] == 'mm', 'Diameter'].apply(correct_Diameter_col)
    df.loc[df['DiameterUnits'] == 'in', 'diameter_mm'] = df.loc[df['DiameterUnits'] == 'in', 'Diameter'].apply(correct_Diameter_col).apply(lambda ds: pd.Series(ds).apply(lambda d: inch_to_mm(float(d))))

    # Convert Bcm/y to MW
    df['CapacityBcm/y'] = pd.to_numeric(df['CapacityBcm/y'], errors='coerce')
    df['capacity [MW]'] = df['CapacityBcm/y'].apply(lambda d: bcm_to_MW(d))

    # Get capacity from diameter for rows where no capacity is given
    df.loc[df['CapacityBcm/y'] == '--', 'capacity [MW]'] = df.loc[df['CapacityBcm/y'] == '--', 'diameter_mm'].apply(lambda d: diameter_to_capacity(int(d)))
    df['diameter_mm'] = pd.to_numeric(df['diameter_mm'], errors='coerce', downcast='integer')
    df.loc[pd.isna(df['CapacityBcm/y']), 'capacity [MW]'] = df.loc[pd.isna(df['CapacityBcm/y']), 'diameter_mm'].apply(lambda d: diameter_to_capacity(d))

    return(df)


def load_IGGIELGN_data(fn):
    df = gpd.read_file(fn)
    param = df.param.apply(pd.Series)
    method = df.method.apply(pd.Series)[["diameter_mm", "max_cap_M_m3_per_d"]]
    method.columns = method.columns + "_method"
    df = pd.concat([df, param, method], axis=1)
    to_drop = ["param", "uncertainty", "method", "tags"]
    to_drop = df.columns.intersection(to_drop)
    df.drop(to_drop, axis=1, inplace=True)
    return df


def prepare_IGGIELGN_data(
    df,
    length_factor=1.5,
    correction_threshold_length=4,
    correction_threshold_p_nom=8,
    bidirectional_below=10,
): # Taken from pypsa-eur and adapted
    # extract start and end from LineString
    df["point0"] = df.geometry.apply(lambda x: Point(x.coords[0]))
    df["point1"] = df.geometry.apply(lambda x: Point(x.coords[-1]))

    conversion_factor = 437.5  # MCM/day to MWh/h
    df["p_nom"] = df.max_cap_M_m3_per_d * conversion_factor

    # for inferred diameters, assume 500 mm rather than 900 mm (more conservative)
    df.loc[df.diameter_mm_method != "raw", "diameter_mm"] = 500.0

    keep = [
        "name",
        "diameter_mm",
        "is_H_gas",
        "is_bothDirection",
        "length_km",
        "p_nom",
        "max_pressure_bar",
        "start_year",
        "point0",
        "point1",
        "geometry",
    ]
    to_rename = {
        "is_bothDirection": "bidirectional",
        "is_H_gas": "H_gas",
        "start_year": "build_year",
        "length_km": "length",
    }
    df = df[keep].rename(columns=to_rename)

    df.bidirectional = df.bidirectional.astype(bool)
    df.H_gas = df.H_gas.astype(bool)

    # short lines below 10 km are assumed to be bidirectional
    short_lines = df["length"] < bidirectional_below
    df.loc[short_lines, "bidirectional"] = True

    # correct all capacities that deviate correction_threshold factor
    # to diameter-based capacities, unless they are NordStream pipelines
    # also all capacities below 0.5 GW are now diameter-based capacities
    df["p_nom_diameter"] = df.diameter_mm.apply(diameter_to_capacity)
    ratio = df.p_nom / df.p_nom_diameter
    not_nordstream = df.max_pressure_bar < 220
    df.p_nom.update(
        df.p_nom_diameter.where(
            (df.p_nom <= 500)
            | ((ratio > correction_threshold_p_nom) & not_nordstream)
            | ((ratio < 1 / correction_threshold_p_nom) & not_nordstream)
        )
    )

    # lines which have way too discrepant line lengths
    # get assigned haversine length * length factor
    df["length_haversine"] = df.apply(
        lambda p: length_factor
        * haversine_pts([p.point0.x, p.point0.y], [p.point1.x, p.point1.y]),
        axis=1,
    )
    ratio = df.eval("length / length_haversine")
    df["length"].update(
        df.length_haversine.where(
            (df["length"] < 20)
            | (ratio > correction_threshold_length)
            | (ratio < 1 / correction_threshold_length)
        )
    )

    # Convert CRS to EPSG:3857 so we can measure distances
    df = df.to_crs(epsg=3857)

    return df
  


def load_bus_region(onshore_path, pipelines):
    """
    Load pypsa-earth-sec onshore regions. 
    TODO: Think about including Offshore regions but only for states that have offshore pipelines.
    """
    bus_regions_onshore = gpd.read_file(onshore_path)
    # Convert CRS to EPSG:3857 so we can measure distances
    bus_regions_onshore = bus_regions_onshore.to_crs(epsg=3857) 

    bus_regions_onshore = bus_regions_onshore.rename(
        {'name':'gadm_id'}, axis=1).loc[:, ['gadm_id', 'geometry']]

    if snakemake.config["clustering_options"]["alternative_clustering"]:
        # Conversion of GADM id to from 3 to 2-digit
        bus_regions_onshore["gadm_id"] = (
            bus_regions_onshore["gadm_id"]
            .str.split(".")
            .apply(lambda id: two_2_three_digits_country(id[0]) + "." + id[1])
        )
        bus_regions_onshore['gadm_id'] = bus_regions_onshore['gadm_id'].str.replace('_AC', '')

    country_borders = unary_union(bus_regions_onshore.geometry)

    # Create a new GeoDataFrame containing the merged polygon
    country_borders = gpd.GeoDataFrame(geometry=[country_borders], crs=pipelines.crs)

    return bus_regions_onshore, country_borders



def get_states_in_order(pipeline, bus_regions_onshore):
    states_p = []

    if pipeline.geom_type == "LineString":
        # Interpolate points along the LineString with a given step size (e.g., 5)
        step_size = 10000
        interpolated_points = [pipeline.interpolate(i) for i in range(0, int(pipeline.length), step_size)]
        interpolated_points.append(pipeline.interpolate(pipeline.length))  # Add the last point

    elif pipeline.geom_type == "MultiLineString":
        interpolated_points=[]
        # Iterate over each LineString within the MultiLineString
        for line in pipeline.geoms:
            # Interpolate points along each LineString with a given step size (e.g., 5)
            step_size = 10000
            interpolated_points_line = [line.interpolate(i) for i in range(0, int(line.length), step_size)]
            interpolated_points_line.append(line.interpolate(line.length))  # Add the last point
            interpolated_points.extend(interpolated_points_line)

    # Check each interpolated point against the state geometries
    for point in interpolated_points:
        for index, state_row in bus_regions_onshore.iterrows():
            if state_row.geometry.contains(point):
                gadm_id = state_row["gadm_id"]
                if gadm_id not in states_p:
                    states_p.append(gadm_id)
                break  # Stop checking other states once a match is found

    return states_p


def parse_states(pipelines, bus_regions_onshore):
    #Parse the states of the points which are connected by the pipeline geometry object
    pipelines["nodes"] = None
    pipelines["states_passed"] = None
    pipelines["amount_states_passed"] = None

    for pipeline, row in pipelines.iterrows():
        states_p = get_states_in_order(row.geometry, bus_regions_onshore)
        # states_p = pd.unique(states_p)
        row['states_passed'] = states_p
        row["amount_states_passed"] = len(states_p)
        row["nodes"] = list(zip(states_p[0::1], states_p[1::1]))
        pipelines.loc[pipeline] = row
    print("The maximum number of states which are passed by one single pipeline amounts to {}.".format(pipelines.states_passed.apply(lambda n: len(n)).max()))
    return pipelines

def cluster_gas_network(pipelines, bus_regions_onshore, length_factor):
    #drop innerstatal pipelines
    pipelines_interstate = pipelines.drop(pipelines.loc[pipelines.amount_states_passed < 2].index)

    # Convert CRS to EPSG:3857 so we can measure distances
    pipelines_interstate = pipelines_interstate.to_crs(epsg=3857)  # 3857

    # Perform overlay operation to split lines by polygons
    pipelines_interstate = gpd.overlay(pipelines_interstate, bus_regions_onshore, how='intersection')

    column_set = ['ProjectID', 'nodes',  'gadm_id', 'capacity [MW]'] 

    if snakemake.config["custom_data"]["gas_grid_dataset"] == 'IGGIELGN':
        pipelines_per_state = pipelines_interstate.rename({'p_nom':'capacity [MW]', 'name':'ProjectID'}, axis=1).loc[:, column_set].reset_index(drop=True)
    elif snakemake.config["custom_data"]["gas_grid_dataset"] == 'GGIT':
        pipelines_per_state = pipelines_interstate.loc[:, column_set].reset_index(drop=True)

    # Explode the column containing lists of tuples
    df_exploded = pipelines_per_state.explode('nodes').reset_index(drop=True)

    # Create new columns for the tuples
    df_exploded.insert(0,"bus1", pd.DataFrame(df_exploded['nodes'].tolist())[1])
    df_exploded.insert(0,"bus0", pd.DataFrame(df_exploded['nodes'].tolist())[0])

    # Drop the original column
    df_exploded.drop('nodes', axis=1, inplace=True)

    # Reset the index if needed
    df_exploded.reset_index(drop=True, inplace=True)

    # Custom function to check if value in column 'gadm_id' exists in either column 'bus0' or column 'bus1'
    def check_existence(row):
        return row['gadm_id'] in [row['bus0'], row['bus1']]

    # Apply the custom function to each row and keep only the rows that satisfy the condition
    df_filtered = df_exploded[df_exploded.apply(check_existence, axis=1)]
    df_grouped = df_filtered.groupby(['bus0', 'bus1', 'ProjectID'], as_index=False).agg({
                                                                    'capacity [MW]':'first',
                                                                    })

    # Rename columns to match pypsa-earth-sec format
    df_grouped = df_grouped.rename({'capacity [MW]':'capacity'}, axis=1).loc[:, ['bus0', 'bus1', 'capacity']]
    # df_exploded = df_exploded.loc[:, ['bus0', 'bus1', 'length']] # 'capacity'

    # Group by buses to get average length and sum of capacites of all pipelines between any two states on the route.
    grouped = df_grouped.groupby(['bus0', 'bus1'], as_index=False).agg({
                                                                    'capacity':'sum'
                                                                    })
    states1 = bus_regions_onshore.copy()
    states1 = states1.set_index('gadm_id')

    # Create center points for each polygon and store them in a new column 'center_point'
    states1['center_point'] = states1['geometry'].to_crs(3857).centroid.to_crs(4326) # ----> If haversine_pts method  for length calc is used
    # states1['center_point'] = states1['geometry'].centroid

    # Create an empty DataFrame to store distances
    distance_data = []

    # Iterate over all combinations of polygons
    for i in range(len(states1)):
        for j in range(len(states1)):
            if i != j:
                polygon1 = states1.iloc[i]
                polygon2 = states1.iloc[j]

                # Calculate Haversine distance
                distance = haversine_pts([Point(polygon1['center_point'].coords[0]).x, Point(polygon1['center_point'].coords[-1]).y], 
                                    [Point(polygon2['center_point'].coords[0]).x, Point(polygon2['center_point'].coords[-1]).y]) # ----> If haversine_pts method  for length calc is used

                # Store the distance along with polygon IDs or other relevant information
                polygon_id1 = states1.index[i]
                polygon_id2 = states1.index[j]
                distance_data.append([polygon_id1, polygon_id2, distance])

    # Create a DataFrame from the distance data
    distance_df = pd.DataFrame(distance_data, columns=['bus0', 'bus1', 'distance'])

    merged_df = pd.merge(grouped, distance_df, on=['bus0', 'bus1'], how='left')

    length_factor=1.25

    merged_df['length'] = merged_df['distance'] * length_factor

    merged_df = merged_df.drop('distance', axis=1)

    merged_df['GWKm'] =  (merged_df['capacity'] / 1000) * merged_df['length']

    return merged_df


def plot_gas_network(pipelines, country_borders, bus_regions_onshore):
    df = pipelines.copy()
    df = gpd.overlay(df, country_borders, how='intersection')

    if snakemake.config["custom_data"]["gas_grid_dataset"] == 'IGGIELGN':
        df = df.rename({'p_nom':'capacity [MW]'}, axis=1)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12, 7)
    bus_regions_onshore.to_crs(epsg=3857).plot(ax=ax, color="white", edgecolor="darkgrey", linewidth=0.5)
    df.loc[(df.amount_states_passed > 1)].to_crs(epsg=3857).plot(
                                                    ax=ax,
                                                    column='capacity [MW]',
                                                    linewidth=2.5,
                                                    #linewidth=df['capacity [MW]'],
                                                    #alpha=0.8,
                                                    categorical=False,
                                                    cmap='viridis_r',
                                                    #legend=True,
                                                    #legend_kwds={'label':'Pipeline capacity [MW]'},
                                                    )

    df.loc[(df.amount_states_passed <= 1)].to_crs(epsg=3857).plot(
                                                    ax=ax,
                                                    column='capacity [MW]',
                                                    linewidth=2.5,
                                                    #linewidth=df['capacity [MW]'],
                                                    alpha=0.5,
                                                    categorical=False,
                                                    # color='darkgrey',
                                                    ls = 'dotted', 
                                                    )

    # # Create custom legend handles for line types
    # line_types = [ 'solid', 'dashed', 'dotted'] # solid
    # legend_handles = [Line2D([0], [0], color='black', linestyle=line_type) for line_type in line_types]

    # Define line types and labels
    line_types = ['solid', 'dotted']
    line_labels = ['Operating', 'Not considered \n(within-state)']

    # Create custom legend handles for line types
    legend_handles = [Line2D([0], [0], color='black', linestyle=line_type, label=line_label)
                    for line_type, line_label in zip(line_types, line_labels)]


    # Add the line type legend
    ax.legend(handles=legend_handles, title='Status',borderpad=1,
                title_fontproperties={'weight':'bold'}, fontsize=11, loc=1,)

    # # create the colorbar
    norm = colors.Normalize(vmin=df['capacity [MW]'].min(), vmax=df['capacity [MW]'].max())
    cbar = plt.cm.ScalarMappable(norm=norm, cmap='viridis_r')
    # fig.colorbar(cbar, ax=ax).set_label('Capacity [MW]')

    # add colorbar
    ax_cbar = fig.colorbar(cbar, ax=ax, location='left', shrink=0.8, pad=0.01)
    # add label for the colorbar
    ax_cbar.set_label('Natural gas pipeline capacity [MW]', fontsize=15)

    ax.set_axis_off()        
    fig.savefig(snakemake.output.gas_network_fig, dpi=300, bbox_inches="tight")




if snakemake.config["custom_data"]["gas_grid"]:
    if snakemake.config["custom_data"]["gas_grid_dataset"] == 'GGIT':

        pipelines = download_GGIT_gas_network()
        pipelines = prepare_GGIT_data(pipelines)

    elif snakemake.config["custom_data"]["gas_grid_dataset"] == 'IGGIELGN':
        
        download_IGGIELGN_gas_network()

        pipelines = load_IGGIELGN_data(snakemake.input.gas_network)
        pipelines = prepare_IGGIELGN_data(pipelines)

    bus_regions_onshore = load_bus_region(snakemake.input.regions_onshore, pipelines)[0]
    country_borders = load_bus_region(snakemake.input.regions_onshore, pipelines)[1]

    pipelines = parse_states(pipelines, bus_regions_onshore)

    plot_gas_network(pipelines, country_borders, bus_regions_onshore)

    pipelines = cluster_gas_network(pipelines, bus_regions_onshore, length_factor=1.25)

    pipelines.to_csv(snakemake.output.clustered_gas_network, index=False)

    average_length = pipelines['length'].mean
    print('average_length = ', average_length)

    total_system_capacity = pipelines['GWKm'].sum()
    print('total_system_capacity = ', total_system_capacity)