# -*- coding: utf-8 -*-

import logging
import os

import numpy as np
import pandas as pd
from helpers import mock_snakemake, sets_path_to_root

logger = logging.getLogger(__name__)

from itertools import product

import geopandas as gpd


def w_avg(df, values, weights):
    d = df[values]
    w = df[weights]
    if w.sum() == 0:
        return 0
    else:
        return (d * w).sum() / w.sum()


def calculate_flh_classes(group):
    """
    Function to calculate flh_classes within each group
    """
    n = len(group)
    if n < 4:
        bins = [i / n for i in range(n + 1)]
        labels = [f"Q{i+3}" for i in range(n)]
    else:
        bins = [0, 0.3, 0.6, 0.9, 1.0]
        labels = ["Q1", "Q2", "Q3", "Q4"]

    group["flh_class"] = pd.qcut(group["flh"], q=bins, labels=labels)
    return group


def load_data(technology):
    df_t = pd.read_csv(snakemake.input[f"{technology}_pot_t"])
    df = pd.read_csv(snakemake.input[f"{technology}_pot"])
    df = df.loc[df["simyear"].isin(snakemake.config["scenario"]["planning_horizons"])]
    df_t = df_t.loc[
        df_t["simyear"].isin(snakemake.config["scenario"]["planning_horizons"])
    ]
    return df, df_t


def merge_onwind(onwind, onwind_rest):
    """
    Returns a merged onwind dataset as a dictionary containing the potentials and hourdata with respective keys.
    The merged dataframes substitute the onwind_rest entries with the onwind entries for the regions where there are onwind entries.
    Furthermore, the additional steps of onwind are added.
    """
    # Create copies of the dataframes
    merged_potentials = onwind["potentials"].copy()
    merged_hourdata = onwind["hourdata"].copy()

    # Identify the indices to drop from onwind_rest
    potentials_index_to_drop = merged_potentials.set_index(
        ["region", "step", "simyear"]
    ).index
    hourdata_index_to_drop = merged_hourdata.set_index(
        ["region", "step", "simyear"]
    ).index

    # Sort by region, simyear, and flh in descending order
    merged_potentials = merged_potentials.sort_values(
        by=["region", "simyear", "flh"], ascending=[True, True, False]
    )
    # Group by region and simyear, then assign a rank starting from zero
    merged_potentials["step_new"] = merged_potentials.groupby(
        ["region", "simyear"]
    ).cumcount()

    # Add additional column step_new for merged_hourdata using the mapping from merged_potentials
    merged_hourdata["step_new"] = merged_hourdata.set_index(
        ["region", "step", "simyear"]
    ).index.map(merged_potentials.set_index(["region", "step", "simyear"])["step_new"])

    # Step_new should be named step and step discarded
    merged_potentials.rename(
        columns={"step": "step_old", "step_new": "step"}, inplace=True
    )
    merged_hourdata.rename(
        columns={"step": "step_old", "step_new": "step"}, inplace=True
    )

    # Drop the old step column
    merged_potentials.drop(columns=["step_old"], inplace=True)
    merged_hourdata.drop(columns=["step_old"], inplace=True)

    # Drop entries in onwind_rest that are already in onwind
    filtered_potentials_rest = onwind_rest["potentials"].loc[
        ~onwind_rest["potentials"]
        .set_index(["region", "step", "simyear"])
        .index.isin(potentials_index_to_drop)
    ]
    filtered_hourdata_rest = onwind_rest["hourdata"].loc[
        ~onwind_rest["hourdata"]
        .set_index(["region", "step", "simyear"])
        .index.isin(hourdata_index_to_drop)
    ]

    # Concatenate the dataframes
    merged_potentials = pd.concat([merged_potentials, filtered_potentials_rest])
    merged_hourdata = pd.concat([merged_hourdata, filtered_hourdata_rest])

    return {"potentials": merged_potentials, "hourdata": merged_hourdata}


def prepare_enertile(df, df_t):
    tech_dict = {
        "windonshore": "onwind",
        "sopv": "solar",
    }

    regions = gpd.read_file(snakemake.input.regions_onshore_elec_s)

    df_t["tech"] = df_t["tech"].map(tech_dict)
    df_t["region"] = df_t["region"].replace({"BRA_": "BR.", "_1$": "_1_AC"}, regex=True)

    df["tech"] = df["tech"].map(tech_dict)
    df["region"] = df["region"].replace({"BRA_": "BR.", "_1$": "_1_AC"}, regex=True)

    # Append missing regions to timeseries data
    missing = list(set(regions.name) - set(df.region.unique()))
    if len(missing) > 0:
        df_t_miss = pd.DataFrame(
            list(product(missing, range(8760))), columns=["region", "hour"]
        )

        df_t_miss["tech"] = technology
        df_t_miss["step"] = pd.Series([df_t.step.unique().tolist()] * len(df_t_miss))
        df_t_miss = df_t_miss.explode("step")
        df_t_miss["value"] = 0

        df_t_miss["simyear"] = pd.Series(
            [df_t.simyear.unique().tolist()] * len(df_t_miss)
        )
        df_t_miss = df_t_miss.explode("simyear")
        df_t = pd.concat([df_t, df_t_miss])

        df_miss = pd.DataFrame(data=missing, columns=["region"])

        df_miss["potstepsizeMW"] = 0
        df_miss["simyear"] = pd.Series([df.simyear.unique().tolist()] * len(df_miss))
        df_miss = df_miss.explode("simyear")
        df_miss["tech"] = technology
        df_miss["step"] = pd.Series([df.step.unique().tolist()] * len(df_miss))
        df_miss = df_miss.explode("step")
        df_miss["flh"] = 0
        df_miss["installedcapacity"] = 0
        df_miss["annualcostEuroPMW"] = df["annualcostEuroPMW"].mean()
        df_miss["variablecostEuroPMWh"] = df["variablecostEuroPMWh"].mean()
        df_miss["investmentEuroPKW"] = df["investmentEuroPKW"].mean()
        df_miss["interestrate"] = df["interestrate"].mean().round(3)
        df_miss["lifetime"] = df["lifetime"].mean()
        df_miss["scenarioid"] = df["scenarioid"].mean()
        df_miss["fixedomEuroPKW"] = df["fixedomEuroPKW"].mean()

        df = pd.concat([df, df_miss])
    df.rename(
        columns={"region": "Generator", "potstepsizeMW": "p_nom_max"}, inplace=True
    )
    df["potential"] = df["flh"] * df["p_nom_max"]
    df = df.groupby(["Generator", "step", "simyear"], as_index=False).mean(
        numeric_only=True
    )

    # if technology == "onwind":
    #     bins = [-float("inf"), 1, 2, float("inf")]
    #     labels = ["very good", "good", "remaining"]
    # else:
    #     bins = [-float("inf"), float("inf")]
    #     labels = ["very good"]

    df = df.groupby(["Generator", "step", "simyear"], as_index=False).mean()
    df = df.groupby(["Generator", "simyear"], as_index=False).apply(
        calculate_flh_classes
    )

    df.set_index(["Generator", "step", "simyear"], inplace=True)
    df_t.set_index(["region", "step", "simyear"], inplace=True)
    df_t["install_cap"] = df["p_nom_max"]
    df_t["potential"] = df["potential"]
    df_t["flh_class"] = df["flh_class"]
    df_t.reset_index(inplace=True)

    df_t.rename(columns={"region": "Generator"}, inplace=True)
    df_t = (
        df_t.groupby(["Generator", "tech", "simyear", "flh_class", "hour"])
        .apply(w_avg, "value", "install_cap")
        .reset_index()
    )
    df_t.rename(columns={0: "value"}, inplace=True)
    res_t = pd.pivot_table(
        df_t,
        values="value",
        index=["hour"],
        columns=["Generator", "simyear", "flh_class"],
    )
    res_t.index = pd.date_range(
        start="01-01-2013 00:00:00", end="31-12-2013 23:00:00", freq="h"
    )

    df.reset_index(inplace=True)

    flh = df.groupby(["Generator", "simyear", "flh_class"]).apply(
        w_avg, "flh", "p_nom_max"
    )
    installable = df.groupby(["Generator", "simyear", "flh_class"]).agg(
        {
            "p_nom_max": "sum",
            "annualcostEuroPMW": "mean",
            "fixedomEuroPKW": "mean",
            "installedcapacity": "sum",
            "lifetime": "mean",
            "interestrate": "mean",
        }
    )

    res_t = res_t.multiply(flh)

    # Export for every simyear and flh_class the sliced installable and res_t
    flh_class_dict = {"Q4": "", "Q3": "2", "Q2": "3", "Q1": "4"}
    for region, simyear, flh_class in installable.index:
        ir = (
            installable.loc[(slice(None), simyear, flh_class)]
            .interestrate.mode()
            .item()
        )
        export_tech = technology + flh_class_dict[flh_class]
        logger.info(snakemake.output.keys())
        installable.loc[(slice(None), simyear, flh_class)].reset_index().to_csv(
            snakemake.output[
                f"{export_tech}_{simyear}_{ir}_installable_s{snakemake.wildcards.simpl}_{snakemake.wildcards.clusters}"
            ],
            index=False,
        )
        res_t.loc[:, (slice(None), simyear, flh_class)].droplevel(
            ["simyear", "flh_class"], axis=1
        ).to_csv(
            snakemake.output[
                f"{export_tech}_{simyear}_{ir}_potential_s{snakemake.wildcards.simpl}_{snakemake.wildcards.clusters}"
            ]
        )


if __name__ == "__main__":
    if "snakemake" not in globals():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        snakemake = mock_snakemake(
            "prepare_res_potentials",
            simpl="",
            clusters="11",
            planning_horizons="2030",
            discountrate=0.071,
        )
        sets_path_to_root("pypsa-earth-sec")

    renewables_enertile = snakemake.config["custom_data"]["renewables_enertile"]
    data = {}
    for technology in renewables_enertile:
        data_tech, data_tech_t = load_data(technology)
        data[technology] = {"potentials": data_tech, "hourdata": data_tech_t}
    onwind_merged = merge_onwind(data["onwind"], data["onwind_rest"])
    data["onwind"] = {
        "potentials": onwind_merged["potentials"],
        "hourdata": onwind_merged["hourdata"],
    }
    renewables_enertile.remove("onwind_rest")
    for technology in renewables_enertile:
        prepare_enertile(
            df=data[technology]["potentials"], df_t=data[technology]["hourdata"]
        )
