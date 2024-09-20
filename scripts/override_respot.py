# -*- coding: utf-8 -*-

import logging
import os
from itertools import dropwhile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pypsa
import pytz
import xarray as xr
from helpers import mock_snakemake, override_component_attrs, sets_path_to_root

logger = logging.getLogger(__name__)


def override_values(tech, year, dr, simpl, clusters):
    custom_res_t = pd.read_csv(
        snakemake.input[
            "custom_res_pot_{0}_{1}_{2}_s{3}_{4}".format(
                tech, year, dr, simpl, clusters
            )
        ],
        index_col=0,
        parse_dates=True,
    ).filter(buses, axis=1)

    custom_res = pd.read_csv(
        snakemake.input[
            "custom_res_ins_{0}_{1}_{2}_s{3}_{4}".format(
                tech, year, dr, simpl, clusters
            )
        ],
    )

    custom_res.index = custom_res["Generator"] + " " + tech
    custom_res_t.columns = custom_res_t.columns + " " + tech

    if tech.replace("-", " ") in n.generators.carrier.unique():
        to_drop = n.generators[n.generators.carrier == tech].index
        custom_res.loc[to_drop, "installedcapacity"] = n.generators.loc[
            to_drop, "p_nom"
        ]
        mask = custom_res["installedcapacity"] > custom_res.p_nom_max
        if mask.any():
            logger.info(
                f"Installed capacities exceed maximum installable capacities for {tech} at nodes {custom_res.loc[mask].index}."
            )
        n.mremove("Generator", to_drop)

    if snakemake.wildcards["planning_horizons"] == 2050:
        directory = "results/" + snakemake.params.run.replace("2050", "2030")
        n_name = snakemake.input.network.split("/")[-1].replace(
            n.config["scenario"]["clusters"], ""
        )
        df = pd.read_csv(directory + "/res_caps_" + n_name, index_col=0)
        # df = pd.read_csv(snakemake.config["custom_data"]["existing_renewables"], index_col=0)
        existing_res = df.loc[tech]
        existing_res.index = existing_res.index.str.apply(lambda x: x + tech)
    else:
        existing_res = custom_res["installedcapacity"].values

    n.madd(
        "Generator",
        custom_res.index,
        bus=custom_res["Generator"],
        carrier=tech,
        p_nom_extendable=True,
        p_nom_max=custom_res["p_nom_max"],
        # weight=ds["weight"].to_pandas(),
        # marginal_cost=custom_res["fixedomEuroPKW"].values * 1000,
        capital_cost=custom_res["annualcostEuroPMW"],
        efficiency=1.0,
        p_max_pu=custom_res_t,
        lifetime=custom_res["lifetime"],
        p_nom=custom_res["installedcapacity"],
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        snakemake = mock_snakemake(
            "override_respot",
            simpl="",
            clusters="11",
            ll="v2.0",
            opts="Co2L",
            planning_horizons="2030",
            sopts="144H",
            demand="AB",
            discountrate=0.071,
        )
        sets_path_to_root("pypsa-earth-sec")

    overrides = override_component_attrs(snakemake.input.overrides)
    n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)
    m = n.copy()
    if snakemake.params.custom_data["renewables"]:
        buses = list(n.buses[n.buses.carrier == "AC"].index)
        energy_totals = pd.read_csv(snakemake.input.energy_totals, index_col=0)
        countries = snakemake.params.countries
        if snakemake.params.custom_data["renewables"]:
            techs = snakemake.params.custom_data["renewables"]
            year = snakemake.wildcards["planning_horizons"]
            dr = snakemake.wildcards["discountrate"]
            simpl = snakemake.wildcards["simpl"]
            clusters = snakemake.wildcards["clusters"]

            m = n.copy()

            for tech in techs:
                override_values(tech, year, dr, simpl, clusters)

        else:
            print("No RES potential techs to override...")

        if snakemake.params.custom_data["elec_demand"]:
            for country in countries:
                n.loads_t.p_set.filter(like=country)[buses] = (
                    (
                        n.loads_t.p_set.filter(like=country)[buses]
                        / n.loads_t.p_set.filter(like=country)[buses].sum().sum()
                    )
                    * energy_totals.loc[country, "electricity residential"]
                    * 1e6
                )

    n.export_to_netcdf(snakemake.output[0])
