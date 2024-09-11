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

    # Append missing regions to timeseries data
    missing = list(set(regions.name) - set(df.region.unique()))
    df_t_miss = pd.DataFrame(
        list(product(missing, range(8760))), columns=["region", "hour"]
    )
    df_t_miss["tech"] = technology
    df_t_miss["step"] = pd.Series([df_t.step.unique().tolist()] * len(df_t_miss))
    df_t_miss = df_t_miss.explode("step")
    df_t_miss["value"] = 0

    df_t_miss["simyear"] = pd.Series([df_t.simyear.unique().tolist()] * len(df_t_miss))
    df_t_miss = df_t_miss.explode("simyear")
    df_t = pd.concat([df_t, df_t_miss])
    df_t.set_index(["region", "step", "simyear"], inplace=True)

    df["tech"] = df["tech"].map(tech_dict)
    df["region"] = df["region"].replace({"BRA_": "BR.", "_1$": "_1_AC"}, regex=True)
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
    df = df.groupby(["Generator", "step", "simyear"]).mean(numeric_only=True)

    if technology == "onwind":
        bins = [-float("inf"), 1, 2, float("inf")]
        labels = ["very good", "good", "remaining"]
    else:
        bins = [-float("inf"), float("inf")]
        labels = ["very good"]

    df_t["install_cap"] = df["p_nom_max"]
    df_t.reset_index(inplace=True)
    df_t["step_class"] = pd.cut(
        df_t["step"],
        bins=bins,
        labels=labels,
    )
    df_t.rename(columns={"region": "Generator"}, inplace=True)
    df_t = (
        df_t.groupby(["Generator", "tech", "simyear", "step_class", "hour"])
        .apply(w_avg, "value", "install_cap")
        .reset_index()
    )
    df_t.rename(columns={0: "value"}, inplace=True)
    res_t = pd.pivot_table(
        df_t,
        values="value",
        index=["hour"],
        columns=["Generator", "simyear", "step_class"],
    )
    res_t.index = pd.date_range(
        start="01-01-2013 00:00:00", end="31-12-2013 23:00:00", freq="h"
    )

    df.reset_index(inplace=True)
    df["step_class"] = pd.cut(
        df["step"],
        bins=bins,
        labels=labels,
    )
    flh = df.groupby(["Generator", "simyear", "step_class"]).apply(
        w_avg, "flh", "p_nom_max"
    )
    installable = df.groupby(["Generator", "simyear", "step_class"]).agg(
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

    # Export for every simyear and step_class the sliced installable and res_t
    step_class_dict = {"very good": "", "good": "2", "remaining": "3"}
    for region, simyear, step_class in installable.index:
        ir = (
            installable.loc[(slice(None), simyear, step_class)]
            .interestrate.mode()
            .item()
        )
        export_tech = technology + step_class_dict[step_class]
        logger.info(snakemake.output.keys())
        installable.loc[(slice(None), simyear, step_class)].reset_index().to_csv(
            snakemake.output[
                f"{export_tech}_{simyear}_{ir}_installable_s{snakemake.wildcards.simpl}_{snakemake.wildcards.clusters}"
            ],
            index=False,
        )
        res_t.loc[:, (slice(None), simyear, step_class)].droplevel(
            ["simyear", "step_class"], axis=1
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
