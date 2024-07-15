# -*- coding: utf-8 -*-
"""Solve network."""
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
from helpers import override_component_attrs
from linopy import merge
from pypsa.optimization.abstract import optimize_transmission_expansion_iteratively
from pypsa.optimization.compat import define_constraints, get_var, linexpr
from pypsa.optimization.optimize import optimize
from vresutils.benchmark import memory_logger

logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)


def add_land_use_constraint(n):
    if "m" in snakemake.wildcards.clusters:
        _add_land_use_constraint_m(n)
    else:
        _add_land_use_constraint(n)


def _add_land_use_constraint(n):
    # warning: this will miss existing offwind which is not classed AC-DC and has carrier 'offwind'

    for carrier in ["solar", "onwind", "offwind-ac", "offwind-dc"]:
        existing = (
            n.generators.loc[n.generators.carrier == carrier, "p_nom"]
            .groupby(n.generators.bus.map(n.buses.location))
            .sum()
        )
        existing.index += " " + carrier + "-" + snakemake.wildcards.planning_horizons
        n.generators.loc[existing.index, "p_nom_max"] -= existing

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def _add_land_use_constraint_m(n):
    # if generators clustering is lower than network clustering, land_use accounting is at generators clusters

    planning_horizons = snakemake.config["scenario"]["planning_horizons"]
    grouping_years = snakemake.config["existing_capacities"]["grouping_years"]
    current_horizon = snakemake.wildcards.planning_horizons

    for carrier in ["solar", "onwind", "offwind-ac", "offwind-dc"]:
        existing = n.generators.loc[n.generators.carrier == carrier, "p_nom"]
        ind = list(
            set(
                [
                    i.split(sep=" ")[0] + " " + i.split(sep=" ")[1]
                    for i in existing.index
                ]
            )
        )

        previous_years = [
            str(y)
            for y in planning_horizons + grouping_years
            if y < int(snakemake.wildcards.planning_horizons)
        ]

        for p_year in previous_years:
            ind2 = [
                i for i in ind if i + " " + carrier + "-" + p_year in existing.index
            ]
            sel_current = [i + " " + carrier + "-" + current_horizon for i in ind2]
            sel_p_year = [i + " " + carrier + "-" + p_year for i in ind2]
            n.generators.loc[sel_current, "p_nom_max"] -= existing.loc[
                sel_p_year
            ].rename(lambda x: x[:-4] + current_horizon)

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def prepare_network(n, solve_opts=None):
    # if snakemake.config["rescale_emissions"]:
    #     pass
    # n.carriers.co2_emissions = n.carriers.co2_emissions * 1e-6
    # n.global_constraints.at["CO2Limit", "constant"] = n.global_constraints.at["CO2Limit", "constant"] * 1e-6
    if "lv_limit" in n.global_constraints.index:
        n.line_volume_limit = n.global_constraints.at["lv_limit", "constant"]
        n.line_volume_limit_dual = n.global_constraints.at["lv_limit", "mu"]

    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,
            n.storage_units_t.inflow,
        ):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    if solve_opts.get("load_shedding"):
        n.add("Carrier", "Load")
        n.madd(
            "Generator",
            n.buses.index,
            " load",
            bus=n.buses.index,
            carrier="load",
            sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=1e2,  # Eur/kWh
            # intersect between macroeconomic and surveybased
            # willingness to pay
            # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
            p_nom=1e9,  # kW
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components():
            # if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if "marginal_cost" in t.df:
                np.random.seed(174)
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (
                    np.random.random(len(t.df)) - 0.5
                )

        for t in n.iterate_components(["Line", "Link"]):
            np.random.seed(123)
            t.df["capital_cost"] += (
                1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)
            ) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    if snakemake.config["foresight"] == "myopic":
        add_land_use_constraint(n)

    return n


def add_battery_constraints(n):
    """
    Add constraint ensuring that charger = discharger, i.e.
    1 * charger_size - efficiency * discharger_size = 0
    """
    if not n.links.p_nom_extendable.any():
        return

    discharger_bool = n.links.index.str.contains("battery discharger")
    charger_bool = n.links.index.str.contains("battery charger")

    dischargers_ext = n.links[discharger_bool].query("p_nom_extendable").index
    chargers_ext = n.links[charger_bool].query("p_nom_extendable").index

    eff = n.links.efficiency[dischargers_ext].values
    lhs = (
        n.model["Link-p_nom"].loc[chargers_ext]
        - n.model["Link-p_nom"].loc[dischargers_ext] * eff
    )

    n.model.add_constraints(lhs == 0, name="Link-charger_ratio")


def add_h2_network_cap(n, cap):
    h2_network = n.links.loc[n.links.carrier == "H2 pipeline"]
    if h2_network.empty:
        return
    h2_network_cap = n.model["Link-p_nom"]
    lhs = (h2_network_cap.loc[h2_network.index] * h2_network.length).sum()
    rhs = cap * 1000
    n.model.add_constraints(lhs <= rhs, name="h2_network_cap")


def H2_export_yearly_constraint(n):
    res = [
        "csp",
        "rooftop-solar",
        "solar",
        "onwind",
        "onwind2",
        "offwind",
        "offwind2",
        "ror",
    ]
    res_index = n.generators.loc[n.generators.carrier.isin(res)].index

    weightings = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(res_index)),
        index=n.snapshots,
        columns=res_index,
    )
    capacity_variable = n.model["Generator-p"]

    # single line sum
    res = (weightings * capacity_variable.loc[res_index]).sum()

    load_ind = n.loads[n.loads.carrier == "AC"].index.intersection(
        n.loads_t.p_set.columns
    )

    load = (
        n.loads_t.p_set[load_ind].sum(axis=1) * n.snapshot_weightings["generators"]
    ).sum()

    h2_export = n.loads.loc["H2 export load"].p_set * 8760

    lhs = res

    include_country_load = snakemake.config["policy_config"]["yearly"][
        "re_country_load"
    ]

    if include_country_load:
        elec_efficiency = (
            n.links.filter(like="Electrolysis", axis=0).loc[:, "efficiency"].mean()
        )
        rhs = (
            h2_export * (1 / elec_efficiency) + load
        )  # 0.7 is approximation of electrloyzer efficiency # TODO obtain value from network
    else:
        rhs = h2_export * (1 / 0.7)

    n.model.add_constraints(lhs >= rhs, name="H2ExportConstraint-RESproduction")


def monthly_constraints(n, n_ref):
    res_techs = [
        "csp",
        "rooftop-solar",
        "solar",
        "onwind",
        "onwind2",
        "offwind",
        "offwind2",
        "ror",
    ]
    allowed_excess = snakemake.config["policy_config"]["hydrogen"]["allowed_excess"]

    res_index = n.generators.loc[n.generators.carrier.isin(res_techs)].index

    weightings = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(res_index)),
        index=n.snapshots,
        columns=res_index,
    )
    capacity_variable = n.model["Generator-p"]

    # single line sum
    res = (weightings * capacity_variable[res_index]).sum(axis=1)
    res = res.groupby(res.index.month).sum()

    link_p = n.model["Link-p"]
    electrolysis = link_p.loc[
        n.links.index[n.links.index.str.contains("H2 Electrolysis")]
    ]

    weightings_electrolysis = pd.DataFrame(
        np.outer(
            n.snapshot_weightings["generators"], [1.0] * len(electrolysis.columns)
        ),
        index=n.snapshots,
        columns=electrolysis.columns,
    )

    elec_input = ((-allowed_excess * weightings_electrolysis) * electrolysis).sum(
        axis=1
    )

    elec_input = elec_input.groupby(elec_input.index.month).sum()

    if snakemake.config["policy_config"]["hydrogen"]["additionality"]:
        res_ref = n_ref.generators_t.p[res_index] * weightings
        res_ref = res_ref.groupby(n_ref.generators_t.p.index.month).sum().sum(axis=1)

        elec_input_ref = (
            n_ref.links_t.p0.loc[
                :, n_ref.links_t.p0.columns.str.contains("H2 Electrolysis")
            ]
            * weightings_electrolysis
        )
        elec_input_ref = (
            -elec_input_ref.groupby(elec_input_ref.index.month).sum().sum(axis=1)
        )

        for i in range(len(res.index)):
            lhs = res.iloc[i] + "\n" + elec_input.iloc[i]
            rhs = res_ref.iloc[i] + elec_input_ref.iloc[i]
            n.model.add_constraints(
                lhs >= rhs, name=f"RESconstraints_{i}-REStarget_{i}"
            )

    else:
        for i in range(len(res.index)):
            lhs = res.iloc[i] + "\n" + elec_input.iloc[i]

            n.model.add_constraints(
                lhs >= 0.0, name=f"RESconstraints_{i}-REStarget_{i}"
            )
    # else:
    #     logger.info("ignoring H2 export constraint as wildcard is set to 0")


def add_chp_constraints(n):
    electric_bool = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("electric")
    )
    heat_bool = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("heat")
    )

    electric = n.links.index[electric_bool]
    heat = n.links.index[heat_bool]

    electric_ext = n.links[electric_bool].query("p_nom_extendable").index
    heat_ext = n.links[heat_bool].query("p_nom_extendable").index

    electric_fix = n.links[electric_bool].query("~p_nom_extendable").index
    heat_fix = n.links[heat_bool].query("~p_nom_extendable").index

    p = n.model["Link-p"]  # dimension: [time, link]

    # output ratio between heat and electricity and top_iso_fuel_line for extendable
    if not electric_ext.empty:
        p_nom = n.model["Link-p_nom"]

        lhs = (
            p_nom.loc[electric_ext]
            * (n.links.p_nom_ratio * n.links.efficiency)[electric_ext].values
            - p_nom.loc[heat_ext] * n.links.efficiency[heat_ext].values
        )
        n.model.add_constraints(lhs == 0, name="chplink-fix_p_nom_ratio")

        rename = {"Link-ext": "Link"}
        lhs = (
            p.loc[:, electric_ext]
            + p.loc[:, heat_ext]
            - p_nom.rename(rename).loc[electric_ext]
        )
        n.model.add_constraints(lhs <= 0, name="chplink-top_iso_fuel_line_ext")

    # top_iso_fuel_line for fixed
    if not electric_fix.empty:
        lhs = p.loc[:, electric_fix] + p.loc[:, heat_fix]
        rhs = n.links.p_nom[electric_fix]
        n.model.add_constraints(lhs <= rhs, name="chplink-top_iso_fuel_line_fix")

    # back-pressure
    if not electric.empty:
        lhs = (
            p.loc[:, heat] * (n.links.efficiency[heat] * n.links.c_b[electric].values)
            - p.loc[:, electric] * n.links.efficiency[electric]
        )
        n.model.add_constraints(lhs <= rhs, name="chplink-backpressure")


def add_co2_sequestration_limit(n, sns):
    co2_stores = n.stores.loc[n.stores.carrier == "co2 stored"].index

    if co2_stores.empty:  # or ("Store", "e") not in n.variables.index:
        return

    vars_final_co2_stored = n.model["Store-e"].loc[sns[-1], co2_stores]

    lhs = (1 * vars_final_co2_stored).sum()
    rhs = (
        n.config["sector"].get("co2_sequestration_potential", 5) * 1e6
    )  # TODO change 200 limit (Europe)

    name = "co2_sequestration_limit"

    n.model.add_constraints(lhs <= rhs, name=f"GlobalConstraint-{name}")


def set_h2_colors(n):
    blue_h2 = n.model["Link-p"].loc[
        n.links.index[n.links.index.str.contains("blue H2")]
    ]

    pink_h2 = n.model["Link-p"].loc[
        n.links.index[n.links.index.str.contains("pink H2")]
    ]

    fuelcell_ind = n.loads[n.loads.carrier == "land transport fuel cell"].index

    other_ind = n.loads[
        (n.loads.carrier == "H2 for industry")
        | (n.loads.carrier == "H2 for shipping")
        | (n.loads.carrier == "H2")
    ].index

    load_fuelcell = (
        n.loads_t.p_set[fuelcell_ind].sum(axis=1) * n.snapshot_weightings["generators"]
    ).sum()

    load_other_h2 = n.loads.loc[other_ind].p_set.sum() * 8760

    load_h2 = load_fuelcell + load_other_h2

    weightings_blue = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(blue_h2.columns)),
        index=n.snapshots,
        columns=blue_h2.columns,
    )

    weightings_pink = pd.DataFrame(
        np.outer(n.snapshot_weightings["generators"], [1.0] * len(pink_h2.columns)),
        index=n.snapshots,
        columns=pink_h2.columns,
    )

    total_blue = (weightings_blue * blue_h2).sum().sum()

    total_pink = (weightings_pink * pink_h2).sum().sum()

    rhs_blue = load_h2 * snakemake.config["sector"]["hydrogen"]["blue_share"]
    rhs_pink = load_h2 * snakemake.config["sector"]["hydrogen"]["pink_share"]

    n.model.add_constraints(total_blue == rhs_blue, name="blue_h2_share")

    n.model.add_constraints(total_pink == rhs_pink, name="pink_h2_share")


def extra_functionality(n, snapshots):
    add_battery_constraints(n)

    if (
        snakemake.config["policy_config"]["hydrogen"]["temporal_matching"]
        == "h2_yearly_matching"
    ):
        if snakemake.config["policy_config"]["hydrogen"]["additionality"] == True:
            logger.info(
                "additionality is currently not supported for yearly constraints, proceeding without additionality"
            )
        logger.info("setting h2 export to yearly greenness constraint")
        H2_export_yearly_constraint(n)

    elif (
        snakemake.config["policy_config"]["hydrogen"]["temporal_matching"]
        == "h2_monthly_matching"
    ):
        if not snakemake.config["policy_config"]["hydrogen"]["is_reference"]:
            logger.info("setting h2 export to monthly greenness constraint")
            monthly_constraints(n, n_ref)
        else:
            logger.info("preparing reference case for additionality constraint")

    elif (
        snakemake.config["policy_config"]["hydrogen"]["temporal_matching"]
        == "no_res_matching"
    ):
        logger.info("no h2 export constraint set")

    else:
        raise ValueError(
            'H2 export constraint is invalid, check config["policy_config"]'
        )

    if snakemake.config["sector"]["hydrogen"]["network"]:
        if snakemake.config["sector"]["hydrogen"]["network_limit"]:
            add_h2_network_cap(
                n, snakemake.config["sector"]["hydrogen"]["network_limit"]
            )

    if snakemake.config["sector"]["hydrogen"]["set_color_shares"]:
        logger.info("setting H2 color mix")
        set_h2_colors(n)

    add_co2_sequestration_limit(n, snapshots)


def solve_network(n, config, solving, **kwargs):
    set_of_options = solving["solver"]["options"]
    cf_solving = solving["options"]

    kwargs["solver_options"] = (
        solving["solver_options"][set_of_options] if set_of_options else {}
    )
    kwargs["solver_name"] = solving["solver"]["name"]
    kwargs["extra_functionality"] = extra_functionality

    if kwargs["solver_name"] == "gurobi":
        logging.getLogger("gurobipy").setLevel(logging.CRITICAL)
    skip_iterations = cf_solving.pop("skip_iterations", False)
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    # add to network for extra_functionality
    n.config = config

    if skip_iterations:
        status, condition = n.optimize(**kwargs)
    else:
        kwargs["track_iterations"] = cf_solving["track_iterations"]
        kwargs["min_iterations"] = cf_solving["min_iterations"]
        kwargs["max_iterations"] = cf_solving["max_iterations"]
        status, condition = n.optimize.optimize_transmission_expansion_iteratively(
            **kwargs
        )

    if status != "ok":
        logger.warning(
            f"Solving status '{status}' with termination condition '{condition}'"
        )
    if "infeasible" in condition:
        labels = n.model.compute_infeasibilities()
        logger.info(f"Labels:\n{labels}")
        n.model.print_infeasibilities()
        raise RuntimeError("Solving status 'infeasible'")

    return n


def add_existing(n):
    if snakemake.wildcards["planning_horizons"] == "2050":
        directory = (
            "results/"
            + "Existing_capacities/"
            + snakemake.config["run"].replace("2050", "2030")
        )
        n_name = (
            snakemake.input.network.split("/")[-1]
            .replace(str(snakemake.config["scenario"]["clusters"][0]), "")
            .replace(str(snakemake.config["costs"]["discountrate"][0]), "")
            .replace("_presec", "")
            .replace(".nc", ".csv")
        )
        df = pd.read_csv(directory + "/electrolyzer_caps_" + n_name, index_col=0)
        existing_electrolyzers = df.p_nom_opt.values

        h2_index = n.links[n.links.carrier == "H2 Electrolysis"].index
        n.links.loc[h2_index, "p_nom_min"] = existing_electrolyzers

        # n_name = snakemake.input.network.split("/")[-1].replace(str(snakemake.config["scenario"]["clusters"][0]), "").\
        #     replace(".nc", ".csv").replace(str(snakemake.config["costs"]["discountrate"][0]), "")
        df = pd.read_csv(directory + "/res_caps_" + n_name, index_col=0)

        for tech in snakemake.config["custom_data"]["renewables"]:
            # df = pd.read_csv(snakemake.config["custom_data"]["existing_renewables"], index_col=0)
            existing_res = df.loc[tech]
            existing_res.index = existing_res.index.str.apply(lambda x: x + tech)
            tech_index = n.generators[n.generators.carrier == tech].index
            n.generators.loc[tech_index, tech] = existing_res


if __name__ == "__main__":
    if "snakemake" not in globals():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        from helpers import mock_snakemake, sets_path_to_root

        snakemake = mock_snakemake(
            "solve_network",
            simpl="",
            clusters="10",
            ll="c1.0",
            opts="Co2L",
            planning_horizons="2030",
            sopts="144H",
            discountrate=0.071,
            demand="AB",
            h2export="10",
        )

        sets_path_to_root("pypsa-earth-sec")

    logging.basicConfig(
        filename=snakemake.log.python, level=snakemake.config["logging_level"]
    )

    tmpdir = snakemake.config["solving"].get("tmpdir")
    if tmpdir is not None:
        Path(tmpdir).mkdir(parents=True, exist_ok=True)
        opts = snakemake.wildcards.opts.split("-")
    solve_opts = snakemake.config["solving"]["options"]

    fn = getattr(snakemake.log, "memory", None)
    with memory_logger(filename=fn, interval=30.0) as mem:
        overrides = override_component_attrs(snakemake.input.overrides)
        n = pypsa.Network(snakemake.input.network, override_component_attrs=overrides)

        if (
            snakemake.config["custom_data"]["add_existing"]
            and snakemake.wildcards.planning_horizons == "2050"
        ):
            add_existing(n)

        if (
            snakemake.config["policy_config"]["hydrogen"]["additionality"]
            and not snakemake.config["policy_config"]["hydrogen"]["is_reference"]
            and snakemake.config["policy_config"]["hydrogen"]["temporal_matching"]
            != "no_res_matching"
        ):
            n_ref_path = snakemake.config["policy_config"]["hydrogen"]["path_to_ref"]
            n_ref = pypsa.Network(n_ref_path)
        else:
            n_ref = None

        n = prepare_network(n, solve_opts)

        n = solve_network(
            n,
            config=snakemake.config,
            solving=snakemake.params.solving,
            log_fn=snakemake.log.solver,
        )

        n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
        n.export_to_netcdf(snakemake.output[0])

    logger.info("Objective function: {}".format(n.objective))
    logger.info("Objective constant: {}".format(n.objective_constant))
    logger.info("Maximum memory usage: {}".format(mem.mem_usage))
