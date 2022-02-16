import os
import pypsa
import pandas as pd
import numpy as np

from helpers import mock_snakemake, prepare_costs, create_network_topology, create_dummy_data

from types import SimpleNamespace
spatial = SimpleNamespace()

def add_hydrogen(n, costs):
    "function to add hydrogen as an energy carrier with its conversion technologies from and to AC"

    n.add("Carrier", "H2")

    n.madd("Bus", nodes + " H2", location=nodes, carrier="H2")

    n.madd(
        "Link",
        nodes + " H2 Electrolysis",
        bus1=nodes + " H2",
        bus0=nodes,
        p_nom_extendable=True,
        carrier="H2 Electrolysis",
        efficiency=costs.at["electrolysis", "efficiency"],
        capital_cost=costs.at["electrolysis", "fixed"],
        lifetime=costs.at["electrolysis", "lifetime"],
    )

    n.madd(
        "Link",
        nodes + " H2 Fuel Cell",
        bus0=nodes + " H2",
        bus1=nodes,
        p_nom_extendable=True,
        carrier="H2 Fuel Cell",
        efficiency=costs.at["fuel cell", "efficiency"],
        # NB: fixed cost is per MWel
        capital_cost=costs.at["fuel cell", "fixed"] *
        costs.at["fuel cell", "efficiency"],
        lifetime=costs.at["fuel cell", "lifetime"],
    )

    cavern_nodes = pd.DataFrame()
    if options['hydrogen_underground_storage']:
        
          h2_salt_cavern_potential = pd.read_csv(snakemake.input.h2_cavern, index_col=0, squeeze=True)
          h2_cavern_ct = h2_salt_cavern_potential[~h2_salt_cavern_potential.isna()]
          cavern_nodes = n.buses[n.buses.country.isin(h2_cavern_ct.index)]

          h2_capital_cost = costs.at["hydrogen storage underground", "fixed"]

          # assumptions: weight storage potential in a country by population
          # TODO: fix with real geographic potentials
          # convert TWh to MWh with 1e6
          h2_pot = h2_cavern_ct.loc[cavern_nodes.country]
          h2_pot.index = cavern_nodes.index
          # h2_pot = h2_pot * cavern_nodes.fraction * 1e6

          n.madd("Store",
            cavern_nodes.index + " H2 Store",
            bus=cavern_nodes.index + " H2",
            e_nom_extendable=True,
            e_nom_max=h2_pot.values,
            e_cyclic=True,
            carrier="H2 Store",
            capital_cost=h2_capital_cost
        )

    # hydrogen stored overground (where not already underground)
    h2_capital_cost = costs.at["hydrogen storage tank incl. compressor", "fixed"]
    nodes_overground = cavern_nodes.index.symmetric_difference(nodes)

    n.madd("Store",
            nodes_overground + " H2 Store",
            bus=nodes_overground + " H2",
            e_nom_extendable=True,
            e_cyclic=True,
            carrier="H2 Store",
            capital_cost=h2_capital_cost
            )

    attrs = ["bus0", "bus1", "length"]
    h2_links = pd.DataFrame(columns=attrs)

    candidates = pd.concat({"lines": n.lines[attrs],
                            "links": n.links.loc[n.links.carrier == "DC", attrs]})

    for candidate in candidates.index:
        buses = [candidates.at[candidate, "bus0"],
                  candidates.at[candidate, "bus1"]]
        buses.sort()
        name = f"H2 pipeline {buses[0]} -> {buses[1]}"
        if name not in h2_links.index:
            h2_links.at[name, "bus0"] = buses[0]
            h2_links.at[name, "bus1"] = buses[1]
            h2_links.at[name, "length"] = candidates.at[candidate, "length"]

    # TODO Add efficiency losses
    n.madd("Link",
            h2_links.index,
            bus0=h2_links.bus0.values + " H2",
            bus1=h2_links.bus1.values + " H2",
            p_min_pu=-1,
            p_nom_extendable=True,
            length=h2_links.length.values,
            capital_cost=costs.at['H2 (g) pipeline',
                                  'fixed'] * h2_links.length.values,
            carrier="H2 pipeline",
            lifetime=costs.at['H2 (g) pipeline', 'lifetime']
            )


def add_co2(n, costs):


    spatial.nodes = nodes

    spatial.co2 = SimpleNamespace()

    if options["co2_network"]:
        spatial.co2.nodes = nodes + " co2 stored"
        spatial.co2.locations = nodes
        spatial.co2.vents = nodes + " co2 vent"
    else:
        spatial.co2.nodes = ["co2 stored"]
        spatial.co2.locations = ["Africa"]
        spatial.co2.vents = ["co2 vent"]

    spatial.co2.df = pd.DataFrame(vars(spatial.co2), index=nodes)

    # minus sign because opposite to how fossil fuels used:
    # CH4 burning puts CH4 down, atmosphere up
    n.add("Carrier", "co2",
          co2_emissions=-1.)

    # this tracks CO2 in the atmosphere
    n.add("Bus",
        "co2 atmosphere",
        location="Africa", #TODO Ignoed by pypsa chck
        carrier="co2"
    )

    # can also be negative
    n.add("Store",
        "co2 atmosphere",
        e_nom_extendable=True,
        e_min_pu=-1,
        carrier="co2",
        bus="co2 atmosphere"
    )

    # this tracks CO2 stored, e.g. underground
    n.madd("Bus",
        spatial.co2.nodes,
        location=spatial.co2.locations,
        carrier="co2 stored"
    )

    n.madd("Store",
        spatial.co2.nodes.str[:-2] + 'age',
        e_nom_extendable=True,
        e_nom_max=np.inf,
        capital_cost=options['co2_sequestration_cost'],
        carrier="co2 stored",
        bus=spatial.co2.nodes
    )

   
    n.madd("Link",
        spatial.co2.vents,
        bus0=spatial.co2.nodes,
        bus1="co2 atmosphere",
        carrier="co2 vent",
        efficiency=1.,
        p_nom_extendable=
        True
    )

    #logger.info("Adding CO2 network.")
    co2_links = create_network_topology(n, "CO2 pipeline ")

    cost_onshore = (1 - co2_links.underwater_fraction) * costs.at['CO2 pipeline', 'fixed'] * co2_links.length
    cost_submarine = co2_links.underwater_fraction * costs.at['CO2 submarine pipeline', 'fixed'] * co2_links.length
    capital_cost = cost_onshore + cost_submarine

    n.madd("Link",
        co2_links.index,
        bus0=co2_links.bus0.values + " co2 stored",
        bus1=co2_links.bus1.values + " co2 stored",
        p_min_pu=-1,
        p_nom_extendable=True,
        length=co2_links.length.values,
        capital_cost=capital_cost.values,
        carrier="CO2 pipeline",
        lifetime=costs.at['CO2 pipeline', 'lifetime'])

    n.madd("Store",
        spatial.co2.nodes,
        e_nom_extendable=True,
        e_nom_max=np.inf,
        capital_cost=options['co2_sequestration_cost'],
        carrier="co2 stored",
        bus=spatial.co2.nodes
    )


    #logger.info("Adding CO2 network.")
    co2_links = create_network_topology(n, "CO2 pipeline ")

    cost_onshore = (1 - co2_links.underwater_fraction) * costs.at['CO2 pipeline', 'fixed'] * co2_links.length
    cost_submarine = co2_links.underwater_fraction * costs.at['CO2 submarine pipeline', 'fixed'] * co2_links.length
    capital_cost = cost_onshore + cost_submarine


# def add_aviation(n, cost):
    
#     all_aviation = ["total international aviation", "total domestic aviation"]
#     p_set = nodal_energy_totals.loc[nodes, all_aviation].sum(axis=1).sum() * 1e6 / 8760

#     n.add("Load",
#         "kerosene for aviation",
#         bus="EU oil",
#         carrier="kerosene for aviation",
#         p_set=p_set
#     )
    
#     co2_release = ["kerosene for aviation"]
#     co2 = n.loads.loc[co2_release, "p_set"].sum() * costs.at["oil", 'CO2 intensity'] / 8760

#     n.add("Load",
#         "oil emissions",
#         bus="co2 atmosphere",
#         carrier="oil emissions",
#         p_set=-co2
#     )

def add_storage(n, costs):
    
    n.add("Carrier", "battery")

    n.madd("Bus",
        nodes + " battery",
        location=nodes,
        carrier="battery"
    )

    n.madd("Store",
        nodes + " battery",
        bus=nodes + " battery",
        e_cyclic=True,
        e_nom_extendable=True,
        carrier="battery",
        capital_cost=costs.at['battery storage', 'fixed'],
        lifetime=costs.at['battery storage', 'lifetime']
    )

    n.madd("Link",
        nodes + " battery charger",
        bus0=nodes,
        bus1=nodes + " battery",
        carrier="battery charger",
        efficiency=costs.at['battery inverter', 'efficiency']**0.5,
        capital_cost=costs.at['battery inverter', 'fixed'],
        p_nom_extendable=True,
        lifetime=costs.at['battery inverter', 'lifetime']
    )

    n.madd("Link",
        nodes + " battery discharger",
        bus0=nodes + " battery",
        bus1=nodes,
        carrier="battery discharger",
        efficiency=costs.at['battery inverter', 'efficiency']**0.5,
        marginal_cost=options['marginal_cost_storage'],
        p_nom_extendable=True,
        lifetime=costs.at['battery inverter', 'lifetime']
    )
    
    
def h2_ch4_conversions(n, costs):
    
    if options['methanation']:

        n.madd("Link",
            spatial.nodes,
            suffix=" Sabatier",
            bus0=nodes + " H2",
            bus1="EU gas",
            bus2=spatial.co2.nodes,
            p_nom_extendable=True,
            carrier="Sabatier",
            efficiency=costs.at["methanation", "efficiency"],
            efficiency2=-costs.at["methanation", "efficiency"] * costs.at['gas', 'CO2 intensity'],
            capital_cost=costs.at["methanation", "fixed"] * costs.at["methanation", "efficiency"],  # costs given per kW_gas
            lifetime=costs.at['methanation', 'lifetime']
        )

    if options['helmeth']:

        n.madd("Link",
            spatial.nodes,
            suffix=" helmeth",
            bus0=nodes,
            bus1="EU gas",
            bus2=spatial.co2.nodes,
            carrier="helmeth",
            p_nom_extendable=True,
            efficiency=costs.at["helmeth", "efficiency"],
            efficiency2=-costs.at["helmeth", "efficiency"] * costs.at['gas', 'CO2 intensity'],
            capital_cost=costs.at["helmeth", "fixed"],
            lifetime=costs.at['helmeth', 'lifetime']
        )


    if options['SMR']:

        n.madd("Link",
            spatial.nodes,
            suffix=" SMR CC",
            bus0="EU gas",
            bus1=nodes + " H2",
            bus2="co2 atmosphere",
            bus3=spatial.co2.nodes,
            p_nom_extendable=True,
            carrier="SMR CC",
            efficiency=costs.at["SMR CC", "efficiency"],
            efficiency2=costs.at['gas', 'CO2 intensity'] * (1 - options["cc_fraction"]),
            efficiency3=costs.at['gas', 'CO2 intensity'] * options["cc_fraction"],
            capital_cost=costs.at["SMR CC", "fixed"],
            lifetime=costs.at['SMR CC', 'lifetime']
        )

        n.madd("Link",
            nodes + " SMR",
            bus0="EU gas",
            bus1=nodes + " H2",
            bus2="co2 atmosphere",
            p_nom_extendable=True,
            carrier="SMR",
            efficiency=costs.at["SMR", "efficiency"],
            efficiency2=costs.at['gas', 'CO2 intensity'],
            capital_cost=costs.at["SMR", "fixed"],
            lifetime=costs.at['SMR', 'lifetime']
        )
        
        
def add_industry(n, costs):

#     print("adding industrial demand")



#     # 1e6 to convert TWh to MWh
#     industrial_demand = pd.read_csv(snakemake.input.industrial_demand, index_col=0) * 1e6
    industrial_demand=create_dummy_data(n, 'industry', '')

#TODO carrier Biomass

################################################## CARRIER = FOSSIL GAS

    n.add("Bus",
        "gas for industry",
        location="EU",
        carrier="gas for industry")

    n.add("Load",
        "gas for industry",
        bus="gas for industry",
        carrier="gas for industry",
        p_set=industrial_demand.loc[nodes, "methane"].sum() / 8760
    )

    n.add("Link",
        "gas for industry",
        bus0="EU gas",
        bus1="gas for industry",
        bus2="co2 atmosphere",
        carrier="gas for industry",
        p_nom_extendable=True,
        efficiency=1.,
        efficiency2=costs.at['gas', 'CO2 intensity']
    )

    n.madd("Link",
        spatial.co2.locations,
        suffix=" gas for industry CC",
        bus0="EU gas",
        bus1="gas for industry",
        bus2="co2 atmosphere",
        bus3=spatial.co2.nodes,
        carrier="gas for industry CC",
        p_nom_extendable=True,
        capital_cost=costs.at["cement capture", "fixed"] * costs.at['gas', 'CO2 intensity'],
        efficiency=0.9,
        efficiency2=costs.at['gas', 'CO2 intensity'] * (1 - costs.at["cement capture", "capture_rate"]),
        efficiency3=costs.at['gas', 'CO2 intensity'] * costs.at["cement capture", "capture_rate"],
        lifetime=costs.at['cement capture', 'lifetime']
    )

#################################################### CARRIER = HYDROGEN
    n.madd("Load",
        nodes,
        suffix=" H2 for industry",
        bus=nodes + " H2",
        carrier="H2 for industry",
        p_set=industrial_demand.loc[nodes, "hydrogen"] / 8760
    )
    
################################################ CARRIER = LIQUID HYDROCARBONS
    n.add("Load",
        "naphtha for industry",
        bus="EU oil",
        carrier="naphtha for industry",
        p_set=industrial_demand.loc[nodes, "naphtha"].sum() / 8760
    )

#     #NB: CO2 gets released again to atmosphere when plastics decay or kerosene is burned
#     #except for the process emissions when naphtha is used for petrochemicals, which can be captured with other industry process emissions
#     #tco2 per hour
    co2_release = ["naphtha for industry"] #TODO kerosene for aviation should be added too but in the right func.
                                            #check land tranport
    co2 = n.loads.loc[co2_release, "p_set"].sum() * costs.at["oil", 'CO2 intensity'] - industrial_demand.loc[nodes, "process emission from feedstock"].sum() / 8760

    n.add("Load",
        "industry oil emissions",
        bus="co2 atmosphere",
        carrier="industry oil emissions",
        p_set=-co2
    )


########################################################### CARIER = HEAT
#     # TODO simplify bus expression
#     n.madd("Load",
#         nodes,
#         suffix=" low-temperature heat for industry",
#         bus=[node + " urban central heat" if node + " urban central heat" in n.buses.index else node + " services urban decentral heat" for node in nodes],
#         carrier="low-temperature heat for industry",
#         p_set=industrial_demand.loc[nodes, "low-temperature heat"] / 8760
#     )


################################################## CARRIER = ELECTRICITY

#     # remove today's industrial electricity demand by scaling down total electricity demand
    for ct in n.buses.country.dropna().unique():
        # TODO map onto n.bus.country
        loads_i = n.loads.index[(n.loads.index.str[:2] == ct) & (n.loads.carrier == "electricity")] # TODO make sure to check this one, should AC have carrier pf "electricity"?
        if n.loads_t.p_set[loads_i].empty: continue
        factor = 1 - industrial_demand.loc[loads_i, "current electricity"].sum() / n.loads_t.p_set[loads_i].sum().sum()
        n.loads_t.p_set[loads_i] *= factor

    n.madd("Load",
        nodes,
        suffix=" industry electricity",
        bus=nodes,
        carrier="industry electricity",
        p_set=industrial_demand.loc[nodes, "electricity"] / 8760
    )

    n.add("Bus",
        "process emissions",
        location="EU",
        carrier="process emissions"
    )

    # this should be process emissions fossil+feedstock
    # then need load on atmosphere for feedstock emissions that are currently going to atmosphere via Link Fischer-Tropsch demand
    n.add("Load",
        "process emissions",
        bus="process emissions",
        carrier="process emissions",
        p_set=-industrial_demand.loc[nodes,["process emission", "process emission from feedstock"]].sum(axis=1).sum() / 8760
    )

    n.add("Link",
        "process emissions",
        bus0="process emissions",
        bus1="co2 atmosphere",
        carrier="process emissions",
        p_nom_extendable=True,
        efficiency=1.
    )

    #assume enough local waste heat for CC
    n.madd("Link",
        spatial.co2.locations,
        suffix=" process emissions CC",
        bus0="process emissions",
        bus1="co2 atmosphere",
        bus2=spatial.co2.nodes,
        carrier="process emissions CC",
        p_nom_extendable=True,
        capital_cost=costs.at["cement capture", "fixed"],
        efficiency=1 - costs.at["cement capture", "capture_rate"],
        efficiency2=costs.at["cement capture", "capture_rate"],
        lifetime=costs.at['cement capture', 'lifetime']
    )



if __name__ == "__main__":
    if "snakemake" not in globals():
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # from helper import mock_snakemake #TODO remove func from here to helper script
        snakemake = mock_snakemake("prepare_sector_network",
                                   simpl="",
                                   clusters="4")
    # TODO add mock_snakemake func

    # TODO fetch from config

    n = pypsa.Network(snakemake.input.network)

    nodes = n.buses.index

    # costs = pd.read_csv( "{}/pypsa-earth-sec/data/costs.csv".format(os.path.dirname(os.getcwd())))

    Nyears = n.snapshot_weightings.generators.sum() / 8760

    costs = prepare_costs(
        snakemake.input.costs,
        snakemake.config["costs"]["USD2013_to_EUR2013"],
        snakemake.config["costs"]["discountrate"],
        Nyears,
        snakemake.config["costs"]["lifetime"],
    )
    # TODO logging

    # TODO fetch options from the config file

    options = {"co2_network": True,
               "co2_sequestration_potential": 200,  #MtCO2/a sequestration potential for Europe
                "co2_sequestration_cost": 10, #EUR/tCO2 for sequestration of CO2
                "hydrogen_underground_storage": True,
                "h2_cavern": True,
                "marginal_cost_storage": 0,
                "methanation": True, 
                "helmeth": True,
                "SMR": True,
                "cc_fraction": 0.9}  
    
    add_hydrogen(n, costs)      #TODO add costs
    
    add_co2(n, costs)      #TODO add costs
    
    add_storage(n, costs)
    
    h2_ch4_conversions(n, costs)
    
    add_industry(n, costs)

    # TODO define spatial (for biomass and co2)

    # TODO changes in case of myopic oversight

    # TODO add co2 tracking function

    # TODO add generation

    # TODO add storage  HERE THE H2 CARRIER IS ADDED IN PYPSA-EUR-SEC

    # TODO add options as in PyPSA-EUR-SEC