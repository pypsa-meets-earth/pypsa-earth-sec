#%%
import os
import requests
import py7zr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import country_converter as coco
import shapely as shp
from pathlib import Path
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
from pyproj import CRS
import re
import zipfile
from pypsa.geo import haversine_pts
from shapely.geometry import Point
from shapely.ops import unary_union

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 70)



#%%

# change current directory
import os
import sys

module_path = os.path.abspath(os.path.join('../../../../pypsa-earth_old')) # To import helpers
if module_path not in sys.path:
    sys.path.append(module_path+"/scripts")
    
from _helpers import sets_path_to_root, three_2_two_digits_country

sets_path_to_root("pypsa-earth")


#%%

# URL to the zip file
url = "https://zenodo.org/record/4767098/files/IGGIELGN.zip?download=1"


#%%

IGGIELGN_gas_pipeline = gpd.read_file(r'/nfs/home/edd32710/projects/HyPAT/Ukraine_old/documentation/notebooks/additions/gas_pipelines/Scigrid/Data/IGGIELGN_PipeSegments.geojson')


#%%

df = IGGIELGN_gas_pipeline.copy()

param = df.param.apply(pd.Series)
method = df.method.apply(pd.Series)[["diameter_mm", "max_cap_M_m3_per_d"]]
method.columns = method.columns + "_method"
df = pd.concat([df, param, method], axis=1)
to_drop = ["param", "uncertainty", "method", "tags"]
to_drop = df.columns.intersection(to_drop)
df.drop(to_drop, axis=1, inplace=True)


#%%

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


#%%

length_factor=1.5
correction_threshold_length=4
correction_threshold_p_nom=8
bidirectional_below=10

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



#%%

pipelines = df.copy()

# Convert CRS to EPSG:3857 so we can measure distances
pipelines = pipelines.to_crs(epsg=3857)


#%%


# states = gpd.read_file('/nfs/home/edd32710/projects/HyPAT/Ukraine/pypsa-earth/resources/shapes/gadm_shapes.geojson', encoding='utf-8')
states = gpd.read_file('/nfs/home/edd32710/projects/HyPAT/Ukraine_old/documentation/notebooks/additions/gas_pipelines/GADM/gadm36_UKR_1.shp')

# Convert CRS to EPSG:3857 so we can measure distances
states = states.to_crs(epsg=3857) 

states = states.rename({'GID_1':'gadm_id', 'NAME_1':'name'}, axis=1).loc[:, ['name', 'gadm_id', 'geometry']]

# states['gadm_id'] = states['gadm_id'].str.replace('UKR', 'UA')

country_borders = unary_union(states.geometry)

# Create a new GeoDataFrame containing the merged polygon
country_borders = gpd.GeoDataFrame(geometry=[country_borders], crs=pipelines.crs)

states.head()


#%%

from shapely.geometry import LineString, MultiLineString

def get_states_in_order(pipeline):
    states_p = []

    if pipeline.geom_type == "LineString":
        # Interpolate points along the LineString with a given step size (e.g., 5)
        step_size = 5000
        interpolated_points = [pipeline.interpolate(i) for i in range(0, int(pipeline.length), step_size)]
        interpolated_points.append(pipeline.interpolate(pipeline.length))  # Add the last point

    elif pipeline.geom_type == "MultiLineString":
        # Iterate over each LineString within the MultiLineString
        for line in pipeline.geoms:
            # Interpolate points along each LineString with a given step size (e.g., 5)
            step_size = 5000
            interpolated_points = [line.interpolate(i) for i in range(0, int(line.length), step_size)]
            interpolated_points.append(line.interpolate(line.length))  # Add the last point

    # Check each interpolated point against the state geometries
    for point in interpolated_points:
        for index, state_row in states.iterrows():
            if state_row.geometry.contains(point):
                gadm_id = state_row["gadm_id"]
                if gadm_id not in states_p:
                    states_p.append(gadm_id)
                break  # Stop checking other states once a match is found

    return states_p


#%%

#Parse the states of the points which are connected by the pipeline geometry object
pipelines["nodes"] = None
pipelines["states_passed"] = None
pipelines["amount_states_passed"] = None

for pipeline, row in pipelines.iterrows():
    states_p = get_states_in_order(row.geometry)
    # states_p = pd.unique(states_p)
    row['states_passed'] = states_p
    row["amount_states_passed"] = len(states_p)
    row["nodes"] = list(zip(states_p[0::1], states_p[1::1]))
    pipelines.loc[pipeline] = row
print("The maximum number of states which are passed by one single pipeline amounts to {}.".format(pipelines.states_passed.apply(lambda n: len(n)).max()))


#%%

#drop innerstatal pipelines
pipelines_interstate = pipelines.drop(pipelines.loc[pipelines.amount_states_passed < 2].index)

# Convert CRS to EPSG:3857 so we can measure distances
pipelines_interstate = pipelines_interstate.to_crs(epsg=3857)  # 3857

# Perform overlay operation to split lines by polygons
pipelines_interstate = gpd.overlay(pipelines_interstate, states, how='intersection') # , keep_geom_type=False


# Calculate length
# pipelines_interstate['length'] = pipelines_interstate.geometry.length / 1000 / 2 #/ pipelines_interstate['amount_states_passed']

pipelines_interstate




#%%

column_set = ['name_1', 'nodes',  'gadm_id', "length" , 'p_nom'] #
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


#%%

df_grouped = df_filtered.groupby(['bus0', 'bus1', 'name_1'], as_index=False).agg({
                                                                'length':'sum',
                                                                'p_nom':'first',
                                                                })
df_grouped


#%%

# Rename columns to match pypsa-earth-sec format
df_grouped = df_grouped.rename({'p_nom':'capacity'}, axis=1).loc[:, ['bus0', 'bus1', 'length', 'capacity']]
# df_exploded = df_exploded.loc[:, ['bus0', 'bus1', 'length']] # 'capacity'

# Group by buses to get average length and sum of capacites of all pipelines between any two states on the route.
grouped = df_grouped.groupby(['bus0', 'bus1'], as_index=False).agg({
                                                                'length':'mean',
                                                                'capacity':'sum'
                                                                })

grouped




#%%

from shapely.geometry import Point

states1 = states.copy()
states1 = states1.set_index('gadm_id')

# Create center points for each polygon and store them in a new column 'center_point'
states1['center_point'] = states1['geometry'].to_crs(3857).centroid.to_crs(4326) # ----> If haversine_pts method  for length calc is used
# states1['center_point'] = states1['geometry'].centroid

# Initialize a list to store adjacent polygon pairs and their distances
adjacent_polygons = []

# Iterate over the GeoDataFrame to find adjacent pairs and calculate their distances
for index, polygon in states1.iterrows():
    neighbors = states1[states1.geometry.touches(polygon['geometry'])].reset_index()
    for _, neighbor in neighbors.iterrows():
        # Calculate distance between the center points of the two polygons
        # distance = polygon['center_point'].distance(neighbor['center_point'])
        distance = haversine_pts([Point(polygon['center_point'].coords[0]).x, Point(polygon['center_point'].coords[-1]).y], 
                                [Point(neighbor['center_point'].coords[0]).x, Point(neighbor['center_point'].coords[-1]).y]) # ----> If haversine_pts method  for length calc is used
        adjacent_polygons.append((index, neighbor.gadm_id, distance))

# Convert the list of adjacent polygon pairs and distances to a DataFrame
distance_df = pd.DataFrame(adjacent_polygons, columns=['bus0', 'bus1', 'distance'])

distance_df['distance'] = distance_df['distance'] #/ 1000 # ----> If haversine_pts method  for length calc is used
# distance_df['distance'] = distance_df['distance'] / 1000

merged_df = pd.merge(grouped, distance_df, on=['bus0', 'bus1'], how='left')


length_factor=1.25

merged_df['length'] = merged_df['distance'] * length_factor

merged_df = merged_df.drop('distance', axis=1)

merged_df['GWKm'] =  (merged_df['capacity'] / 1000) * merged_df['length']

merged_df.to_csv('/nfs/home/edd32710/projects/HyPAT/Ukraine_old/documentation/notebooks/additions/Plots/existing_infrastructure/gas_network/outputs/pipelines_IGGIELGN.csv', index=False)
merged_df


#%%

average_length = merged_df['length'].mean
print(average_length)

total_system_capacity = merged_df['GWKm'].sum()
print(total_system_capacity)


#%%

# PLOT

pipelines = gpd.overlay(pipelines, country_borders, how='intersection')

from matplotlib.lines import Line2D



#plot pipelines
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(12, 7)
states.to_crs(epsg=3857).plot(ax=ax, color="white", edgecolor="darkgrey", linewidth=0.5)
pipelines.loc[(pipelines.amount_states_passed > 1)].to_crs(epsg=3857).plot(
                                                ax=ax,
                                                column='p_nom',
                                                linewidth=2.5,
                                                #linewidth=pipelines['capacity [MW]'],
                                                #alpha=0.8,
                                                categorical=False,
                                                cmap='viridis_r',
                                                #legend=True,
                                                #legend_kwds={'label':'Pipeline capacity [MW]'},
                                                )

pipelines.loc[(pipelines.amount_states_passed <= 1)].to_crs(epsg=3857).plot(
                                                ax=ax,
                                                column='p_nom',
                                                linewidth=2.5,
                                                #linewidth=pipelines['capacity [MW]'],
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
import matplotlib.colors as colors

norm = colors.Normalize(vmin=pipelines['p_nom'].min(), vmax=pipelines['p_nom'].max())
cbar = plt.cm.ScalarMappable(norm=norm, cmap='viridis_r')
# fig.colorbar(cbar, ax=ax).set_label('Capacity [MW]')

# add colorbar
ax_cbar = fig.colorbar(cbar, ax=ax, location='left', shrink=0.8, pad=0.01)
# add label for the colorbar
ax_cbar.set_label('Natural gas pipeline capacity [MW]', fontsize=15)

                                                

ax.set_axis_off()        
# fig.savefig('/nfs/home/edd32710/projects/HyPAT/Ukraine_old/documentation/notebooks/additions/Plots/existing_infrastructure/existing_gas_pipelines_UA.png', dpi=300, bbox_inches="tight")
