from os.path import exists
from shutil import copyfile

from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider

HTTP = HTTPRemoteProvider()

if not exists("config.yaml"):
    copyfile("config.default.yaml", "config.yaml")


configfile: "config.yaml"


PYPSAEARTH_FOLDER = "./pypsa-earth"


SDIR = config["summary_dir"] + config["run"]
RDIR = config["results_dir"] + config["run"]
CDIR = config["costs_dir"]

CUTOUTS_PATH = (
    "cutouts/cutout-2013-era5-tutorial.nc"
    if config["tutorial"]
    else "cutouts/cutout-2013-era5.nc"
)


wildcard_constraints:
    ll="[a-z0-9\.]+",
    simpl="[a-zA-Z0-9]*|all",
    clusters="[0-9]+m?|all",
    opts="[-+a-zA-Z0-9\.\s]*",
    sopts="[-+a-zA-Z0-9\.\s]*",
    discountrate="[-+a-zA-Z0-9\.\s]*",
    demand="[-+a-zA-Z0-9\.\s]*",
    h2export="[0-9]+m?|all",


subworkflow pypsaearth:
    workdir:
        PYPSAEARTH_FOLDER
    snakefile:
        PYPSAEARTH_FOLDER + "/Snakefile"
    configfile:
        "./config.pypsa-earth.yaml"


rule prepare_sector_networks:
    input:
        expand(
            RDIR
            + "/prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}.nc",
            **config["scenario"],
            **config["costs"]
        ),


rule override_res_all_nets:
    input:
        expand(
            RDIR
            + "/prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_presec.nc",
            **config["scenario"],
            **config["costs"],
            **config["export"]
        ),


rule solve_all_networks:
    input:
        expand(
            RDIR
            + "/postnetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export.nc",
            **config["scenario"],
            **config["costs"],
            **config["export"]
        ),


rule prepare_ports:
    output:
        ports="data/ports.csv",
        # TODO move from data to resources
    script:
        "scripts/prepare_ports.py"


rule prepare_airports:
    output:
        ports="data/airports.csv",
        # TODO move from data to resources
    script:
        "scripts/prepare_airports.py"


rule prepare_gas_network:
    input:
        gas_network="data/gas_network/scigrid-gas/data/IGGIELGN_PipeSegments.geojson",
        # regions_onshore='/nfs/home/edd32710/projects/HyPAT/Ukraine_old/documentation/notebooks/additions/Plots/existing_infrastructure/GADM/gadm36_UKR_1.shp'
        regions_onshore=pypsaearth(
            "resources/bus_regions/regions_onshore_elec_s{simpl}_{clusters}.geojson"
        ),
    output:
        clustered_gas_network="resources/gas_networks/gas_network_elec_s{simpl}_{clusters}.csv",
        gas_network_fig="resources/gas_networks/existing_gas_pipelines_{simpl}_{clusters}.png",
    script:
        "scripts/prepare_gas_network.py"


rule prepare_sector_network:
    input:
        network=RDIR
        + "/prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_presec.nc",
        costs=CDIR + "costs_{planning_horizons}.csv",
        h2_cavern="data/hydrogen_salt_cavern_potentials.csv",
        nodal_energy_totals="resources/demand/heat/nodal_energy_heat_totals_{demand}_s{simpl}_{clusters}_{planning_horizons}.csv",
        transport="resources/demand/transport_{demand}_s{simpl}_{clusters}_{planning_horizons}.csv",
        avail_profile="resources/pattern_profiles/avail_profile_{demand}_s{simpl}_{clusters}_{planning_horizons}.csv",
        dsm_profile="resources/pattern_profiles/dsm_profile_{demand}_s{simpl}_{clusters}_{planning_horizons}.csv",
        nodal_transport_data="resources/demand/nodal_transport_data_{demand}_s{simpl}_{clusters}_{planning_horizons}.csv",
        overrides="data/override_component_attrs",
        clustered_pop_layout="resources/population_shares/pop_layout_elec_s{simpl}_{clusters}.csv",
        industrial_demand="resources/demand/industrial_energy_demand_per_node_elec_s{simpl}_{clusters}_{planning_horizons}_{demand}.csv",
        energy_totals="data/energy_totals_{demand}_{planning_horizons}.csv",
        airports="data/airports.csv",
        ports="data/ports.csv",
        heat_demand="resources/demand/heat/heat_demand_{demand}_s{simpl}_{clusters}_{planning_horizons}.csv",
        ashp_cop="resources/demand/heat/ashp_cop_{demand}_s{simpl}_{clusters}_{planning_horizons}.csv",
        gshp_cop="resources/demand/heat/gshp_cop_{demand}_s{simpl}_{clusters}_{planning_horizons}.csv",
        solar_thermal="resources/demand/heat/solar_thermal_{demand}_s{simpl}_{clusters}_{planning_horizons}.csv",
        district_heat_share="resources/demand/heat/district_heat_share_{demand}_s{simpl}_{clusters}_{planning_horizons}.csv",
        biomass_potentials="data/temp_hard_coded/biomass_potentials_s_37.csv",
        biomass_transport_costs="data/temp_hard_coded/biomass_transport_costs.csv",
        shapes_path=pypsaearth(
            "resources/bus_regions/regions_onshore_elec_s{simpl}_{clusters}.geojson"
        ),
        pipelines="resources/gas_networks/gas_network_elec_s{simpl}_{clusters}.csv",
    output:
        RDIR
        + "/prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}.nc",
    threads: 1
    resources:
        mem_mb=2000,
    benchmark:
        (
            RDIR
            + "/benchmarks/prepare_network/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}"
        )
    script:
        "scripts/prepare_sector_network.py"


rule build_ship_profile:
    output:
        ship_profile="resources/ship_profile_{h2export}TWh.csv",
    script:
        "scripts/build_ship_profile.py"


rule add_export:
    input:
        overrides="data/override_component_attrs",
        export_ports="data/export_ports.csv",
        costs=CDIR + "costs_{planning_horizons}.csv",
        ship_profile="resources/ship_profile_{h2export}TWh.csv",
        network=RDIR
        + "/prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}.nc",
        shapes_path=pypsaearth(
            "resources/bus_regions/regions_onshore_elec_s{simpl}_{clusters}.geojson"
        ),
    output:
        RDIR
        + "/prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export.nc",
        # TODO output file name must be adjusted and integrated in workflow
    script:
        "scripts/add_export.py"


rule override_respot:
    input:
        **{
            f"custom_res_pot_{tech}_{planning_horizons}_{discountrate}": f"resources/custom_renewables/{tech}_{planning_horizons}_{discountrate}_potential.csv"
            for tech in config["custom_data"]["renewables"]
            for discountrate in config["costs"]["discountrate"]
            for planning_horizons in config["scenario"]["planning_horizons"]
        },
        **{
            f"custom_res_ins_{tech}_{planning_horizons}_{discountrate}": f"resources/custom_renewables/{tech}_{planning_horizons}_{discountrate}_installable.csv"
            for tech in config["custom_data"]["renewables"]
            for discountrate in config["costs"]["discountrate"]
            for planning_horizons in config["scenario"]["planning_horizons"]
        },
        overrides="data/override_component_attrs",
        network=pypsaearth("networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc"),
        energy_totals="data/energy_totals_{demand}_{planning_horizons}.csv",
    output:
        RDIR
        + "/prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_presec.nc",
    script:
        "scripts/override_respot.py"


rule prepare_transport_data:
    input:
        network=pypsaearth("networks/elec_s{simpl}_{clusters}.nc"),
        energy_totals_name="data/energy_totals_{demand}_{planning_horizons}.csv",
        traffic_data_KFZ="data/emobility/KFZ__count",
        traffic_data_Pkw="data/emobility/Pkw__count",
        transport_name="resources/transport_data.csv",
        clustered_pop_layout="resources/population_shares/pop_layout_elec_s{simpl}_{clusters}.csv",
        temp_air_total="resources/temperatures/temp_air_total_elec_s{simpl}_{clusters}.nc",
    output:
        # nodal_energy_totals="resources/nodal_energy_totals_s{simpl}_{clusters}.csv",
        transport="resources/demand/transport_{demand}_s{simpl}_{clusters}_{planning_horizons}.csv",
        avail_profile="resources/pattern_profiles/avail_profile_{demand}_s{simpl}_{clusters}_{planning_horizons}.csv",
        dsm_profile="resources/pattern_profiles/dsm_profile_{demand}_s{simpl}_{clusters}_{planning_horizons}.csv",
        nodal_transport_data="resources/demand/nodal_transport_data_{demand}_s{simpl}_{clusters}_{planning_horizons}.csv",
    script:
        "scripts/prepare_transport_data.py"


rule build_cop_profiles:
    input:
        temp_soil_total="resources/temperatures/temp_soil_total_elec_s{simpl}_{clusters}.nc",
        temp_soil_rural="resources/temperatures/temp_soil_rural_elec_s{simpl}_{clusters}.nc",
        temp_soil_urban="resources/temperatures/temp_soil_urban_elec_s{simpl}_{clusters}.nc",
        temp_air_total="resources/temperatures/temp_air_total_elec_s{simpl}_{clusters}.nc",
        temp_air_rural="resources/temperatures/temp_air_rural_elec_s{simpl}_{clusters}.nc",
        temp_air_urban="resources/temperatures/temp_air_urban_elec_s{simpl}_{clusters}.nc",
    output:
        cop_soil_total="resources/cops/cop_soil_total_elec_s{simpl}_{clusters}.nc",
        cop_soil_rural="resources/cops/cop_soil_rural_elec_s{simpl}_{clusters}.nc",
        cop_soil_urban="resources/cops/cop_soil_urban_elec_s{simpl}_{clusters}.nc",
        cop_air_total="resources/cops/cop_air_total_elec_s{simpl}_{clusters}.nc",
        cop_air_rural="resources/cops/cop_air_rural_elec_s{simpl}_{clusters}.nc",
        cop_air_urban="resources/cops/cop_air_urban_elec_s{simpl}_{clusters}.nc",
    resources:
        mem_mb=20000,
    benchmark:
        "benchmarks/build_cop_profiles/s{simpl}_{clusters}"
    script:
        "scripts/build_cop_profiles.py"


rule prepare_heat_data:
    input:
        network=pypsaearth("networks/elec_s{simpl}_{clusters}.nc"),
        energy_totals_name="data/energy_totals_{demand}_{planning_horizons}.csv",
        clustered_pop_layout="resources/population_shares/pop_layout_elec_s{simpl}_{clusters}.csv",
        temp_air_total="resources/temperatures/temp_air_total_elec_s{simpl}_{clusters}.nc",
        cop_soil_total="resources/cops/cop_soil_total_elec_s{simpl}_{clusters}.nc",
        cop_air_total="resources/cops/cop_air_total_elec_s{simpl}_{clusters}.nc",
        solar_thermal_total="resources/demand/heat/solar_thermal_total_elec_s{simpl}_{clusters}.nc",
        heat_demand_total="resources/demand/heat/heat_demand_total_elec_s{simpl}_{clusters}.nc",
        heat_profile="data/heat_load_profile_BDEW.csv",
    output:
        nodal_energy_totals="resources/demand/heat/nodal_energy_heat_totals_{demand}_s{simpl}_{clusters}_{planning_horizons}.csv",
        heat_demand="resources/demand/heat/heat_demand_{demand}_s{simpl}_{clusters}_{planning_horizons}.csv",
        ashp_cop="resources/demand/heat/ashp_cop_{demand}_s{simpl}_{clusters}_{planning_horizons}.csv",
        gshp_cop="resources/demand/heat/gshp_cop_{demand}_s{simpl}_{clusters}_{planning_horizons}.csv",
        solar_thermal="resources/demand/heat/solar_thermal_{demand}_s{simpl}_{clusters}_{planning_horizons}.csv",
        district_heat_share="resources/demand/heat/district_heat_share_{demand}_s{simpl}_{clusters}_{planning_horizons}.csv",
    script:
        "scripts/prepare_heat_data.py"


rule build_base_energy_totals:
    input:
        unsd_paths="data/demand/unsd/paths/Energy_Statistics_Database.xlsx",
    output:
        energy_totals_base="data/energy_totals_base.csv",
    script:
        "scripts/build_base_energy_totals.py"


rule prepare_energy_totals:
    input:
        unsd_paths="data/energy_totals_base.csv",
    output:
        energy_totals="data/energy_totals_{demand}_{planning_horizons}.csv",
    script:
        "scripts/prepare_energy_totals.py"


rule build_solar_thermal_profiles:
    input:
        pop_layout_total="resources/population_shares/pop_layout_total.nc",
        pop_layout_urban="resources/population_shares/pop_layout_urban.nc",
        pop_layout_rural="resources/population_shares/pop_layout_rural.nc",
        regions_onshore=pypsaearth(
            "resources/bus_regions/regions_onshore_elec_s{simpl}_{clusters}.geojson"
        ),
        cutout=pypsaearth(CUTOUTS_PATH),
    output:
        solar_thermal_total="resources/demand/heat/solar_thermal_total_elec_s{simpl}_{clusters}.nc",
        solar_thermal_urban="resources/demand/heat/solar_thermal_urban_elec_s{simpl}_{clusters}.nc",
        solar_thermal_rural="resources/demand/heat/solar_thermal_rural_elec_s{simpl}_{clusters}.nc",
    resources:
        mem_mb=20000,
    benchmark:
        "benchmarks/build_solar_thermal_profiles/s{simpl}_{clusters}"
    script:
        "scripts/build_solar_thermal_profiles.py"


rule build_population_layouts:
    input:
        nuts3_shapes=pypsaearth("resources/shapes/gadm_shapes.geojson"),
        urban_percent="data/urban_percent.csv",
        cutout=pypsaearth(CUTOUTS_PATH),
    output:
        pop_layout_total="resources/population_shares/pop_layout_total.nc",
        pop_layout_urban="resources/population_shares/pop_layout_urban.nc",
        pop_layout_rural="resources/population_shares/pop_layout_rural.nc",
    resources:
        mem_mb=20000,
    benchmark:
        "benchmarks/build_population_layouts"
    threads: 8
    script:
        "scripts/build_population_layouts.py"


rule move_hardcoded_files_temp:
    input:
        "data/temp_hard_coded/energy_totals.csv",
        "data/temp_hard_coded/transport_data.csv",
    output:
        "resources/energy_totals.csv",
        "resources/transport_data.csv",
    shell:
        "cp -a data/temp_hard_coded/. resources"


rule build_clustered_population_layouts:
    input:
        pop_layout_total="resources/population_shares/pop_layout_total.nc",
        pop_layout_urban="resources/population_shares/pop_layout_urban.nc",
        pop_layout_rural="resources/population_shares/pop_layout_rural.nc",
        regions_onshore=pypsaearth(
            "resources/bus_regions/regions_onshore_elec_s{simpl}_{clusters}.geojson"
        ),
        cutout=pypsaearth(CUTOUTS_PATH),
    output:
        clustered_pop_layout="resources/population_shares/pop_layout_elec_s{simpl}_{clusters}.csv",
    resources:
        mem_mb=10000,
    benchmark:
        "benchmarks/build_clustered_population_layouts/s{simpl}_{clusters}"
    script:
        "scripts/build_clustered_population_layouts.py"


rule build_heat_demand:
    input:
        pop_layout_total="resources/population_shares/pop_layout_total.nc",
        pop_layout_urban="resources/population_shares/pop_layout_urban.nc",
        pop_layout_rural="resources/population_shares/pop_layout_rural.nc",
        regions_onshore=pypsaearth(
            "resources/bus_regions/regions_onshore_elec_s{simpl}_{clusters}.geojson"
        ),
        cutout=pypsaearth(CUTOUTS_PATH),
    output:
        heat_demand_urban="resources/demand/heat/heat_demand_urban_elec_s{simpl}_{clusters}.nc",
        heat_demand_rural="resources/demand/heat/heat_demand_rural_elec_s{simpl}_{clusters}.nc",
        heat_demand_total="resources/demand/heat/heat_demand_total_elec_s{simpl}_{clusters}.nc",
    resources:
        mem_mb=20000,
    benchmark:
        "benchmarks/build_heat_demand/s{simpl}_{clusters}"
    script:
        "scripts/build_heat_demand.py"


rule build_temperature_profiles:
    input:
        pop_layout_total="resources/population_shares/pop_layout_total.nc",
        pop_layout_urban="resources/population_shares/pop_layout_urban.nc",
        pop_layout_rural="resources/population_shares/pop_layout_rural.nc",
        regions_onshore=pypsaearth(
            "resources/bus_regions/regions_onshore_elec_s{simpl}_{clusters}.geojson"
        ),
        cutout=pypsaearth(CUTOUTS_PATH),
    output:
        temp_soil_total="resources/temperatures/temp_soil_total_elec_s{simpl}_{clusters}.nc",
        temp_soil_rural="resources/temperatures/temp_soil_rural_elec_s{simpl}_{clusters}.nc",
        temp_soil_urban="resources/temperatures/temp_soil_urban_elec_s{simpl}_{clusters}.nc",
        temp_air_total="resources/temperatures/temp_air_total_elec_s{simpl}_{clusters}.nc",
        temp_air_rural="resources/temperatures/temp_air_rural_elec_s{simpl}_{clusters}.nc",
        temp_air_urban="resources/temperatures/temp_air_urban_elec_s{simpl}_{clusters}.nc",
    resources:
        mem_mb=20000,
    benchmark:
        "benchmarks/build_temperature_profiles/s{simpl}_{clusters}"
    script:
        "scripts/build_temperature_profiles.py"


rule copy_config:
    output:
        SDIR + "/configs/config.yaml",
    threads: 1
    resources:
        mem_mb=1000,
    benchmark:
        SDIR + "/benchmarks/copy_config"
    script:
        "scripts/copy_config.py"


rule copy_commit:
    output:
        SDIR + "/commit_info.txt",
    shell:
        """
        git log -n 1 --pretty=format:"Commit: %H%nAuthor: %an <%ae>%nDate: %ad%nMessage: %s" > {output}
        """


rule solve_network:
    input:
        overrides="data/override_component_attrs",
        # network=RDIR
        # + "/prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}.nc",
        network=RDIR
        + "/prenetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export.nc",
        costs=CDIR + "costs_{planning_horizons}.csv",
        configs=SDIR + "/configs/config.yaml",  # included to trigger copy_config rule
        commit=SDIR + "/commit_info.txt",
    output:
        RDIR
        + "/postnetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export.nc",
    shadow:
        "shallow"
    log:
        solver=RDIR
        + "/logs/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export_solver.log",
        python=RDIR
        + "/logs/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export_python.log",
        memory=RDIR
        + "/logs/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export_memory.log",
    threads: 25
    resources:
        mem_mb=config["solving"]["mem"],
    benchmark:
        (
            RDIR
            + "/benchmarks/solve_network/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export"
        )
    script:
        "scripts/solve_network.py"


rule make_summary:
    input:
        overrides="data/override_component_attrs",
        networks=expand(
            RDIR
            + "/postnetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export.nc",
            **config["scenario"],
            **config["costs"],
            **config["export"]
        ),
        costs=CDIR + "costs_{}.csv".format(config["scenario"]["planning_horizons"][0]),
        plots=expand(
            RDIR
            + "/maps/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}-costs-all_{planning_horizons}_{discountrate}_{demand}_{h2export}export.pdf",
            **config["scenario"],
            **config["costs"],
            **config["export"]
        ),
    output:
        nodal_costs=SDIR + "/csvs/nodal_costs.csv",
        nodal_capacities=SDIR + "/csvs/nodal_capacities.csv",
        nodal_cfs=SDIR + "/csvs/nodal_cfs.csv",
        cfs=SDIR + "/csvs/cfs.csv",
        costs=SDIR + "/csvs/costs.csv",
        capacities=SDIR + "/csvs/capacities.csv",
        curtailment=SDIR + "/csvs/curtailment.csv",
        energy=SDIR + "/csvs/energy.csv",
        supply=SDIR + "/csvs/supply.csv",
        supply_energy=SDIR + "/csvs/supply_energy.csv",
        prices=SDIR + "/csvs/prices.csv",
        weighted_prices=SDIR + "/csvs/weighted_prices.csv",
        market_values=SDIR + "/csvs/market_values.csv",
        price_statistics=SDIR + "/csvs/price_statistics.csv",
        metrics=SDIR + "/csvs/metrics.csv",
    threads: 2
    resources:
        mem_mb=10000,
    benchmark:
        SDIR + "/benchmarks/make_summary"
    script:
        "scripts/make_summary.py"


rule plot_network:
    input:
        overrides="data/override_component_attrs",
        network=RDIR
        + "/postnetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export.nc",
    output:
        map=RDIR
        + "/maps/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}-costs-all_{planning_horizons}_{discountrate}_{demand}_{h2export}export.pdf",
    threads: 2
    resources:
        mem_mb=10000,
    benchmark:
        (
            RDIR
            + "/benchmarks/plot_network/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export"
        )
    script:
        "scripts/plot_network.py"


rule plot_summary:
    input:
        costs=SDIR + "/csvs/costs.csv",
        energy=SDIR + "/csvs/energy.csv",
        balances=SDIR + "/csvs/supply_energy.csv",
    output:
        costs=SDIR + "/graphs/costs.pdf",
        energy=SDIR + "/graphs/energy.pdf",
        balances=SDIR + "/graphs/balances-energy.pdf",
    threads: 2
    resources:
        mem_mb=10000,
    benchmark:
        SDIR + "/benchmarks/plot_summary"
    script:
        "scripts/plot_summary.py"


rule prepare_db:
    input:
        network=RDIR
        + "/postnetworks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export.nc",
    output:
        db=RDIR
        + "/summaries/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}-costs-all_{planning_horizons}_{discountrate}_{demand}_{h2export}export.csv",
    threads: 2
    resources:
        mem_mb=10000,
    benchmark:
        (
            RDIR
            + "/benchmarks/prepare_db/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}_{sopts}_{planning_horizons}_{discountrate}_{demand}_{h2export}export"
        )
    script:
        "scripts/prepare_db.py"


rule run_test:
    run:
        import yaml

        with open(PYPSAEARTH_FOLDER + "/config.tutorial.yaml") as file:
            config_pypsaearth = yaml.full_load(file)
            config_pypsaearth["retrieve_databundle"] = {"show_progress": False}
            config_pypsaearth["electricity"]["extendable_carriers"]["Store"] = []
            config_pypsaearth["electricity"]["extendable_carriers"]["Link"] = []
            config_pypsaearth["electricity"]["co2limit"] = 7.75e7

            with open("./config.pypsa-earth.yaml", "w") as wfile:
                yaml.dump(config_pypsaearth, wfile)

        shell("cp test/config.test1.yaml config.yaml")
        shell("snakemake --cores all solve_all_networks --forceall")


rule clean:
    run:
        shell("rm -r " + PYPSAEARTH_FOLDER + "/resources")
        shell("rm -r " + PYPSAEARTH_FOLDER + "/networks")


if config["custom_data"].get("industry_demand", False) == True:

    rule build_industrial_distribution_key:  #custom data
        input:
            regions_onshore=pypsaearth(
                "resources/bus_regions/regions_onshore_elec_s{simpl}_{clusters}.geojson"
            ),
            clustered_pop_layout="resources/population_shares/pop_layout_elec_s{simpl}_{clusters}.csv",
            industrial_database="resources/custom_data/industrial_database.csv",
            #shapes_path=pypsaearth("resources/bus_regions/regions_onshore_elec_s{simpl}_{clusters}.geojson")
            shapes_path=PYPSAEARTH_FOLDER + "/resources/shapes/MAR2.geojson",
        output:
            industrial_distribution_key="resources/demand/industrial_distribution_key_elec_s{simpl}_{clusters}.csv",
        threads: 1
        resources:
            mem_mb=1000,
        benchmark:
            "benchmarks/build_industrial_distribution_key/s{simpl}_{clusters}"
        script:
            "scripts/build_industrial_distribution_key.py"

    rule build_industry_demand:  #custom data
        input:
            industry_sector_ratios="resources/custom_data/industry_sector_ratios_{demand}_{planning_horizons}.csv",
            industrial_distribution_key="resources/demand/industrial_distribution_key_elec_s{simpl}_{clusters}.csv",
            industrial_production_per_country_tomorrow="resources/custom_data/industrial_production_per_country_tomorrow_{planning_horizons}_{demand}.csv",
            costs=CDIR
            + "costs_{}.csv".format(config["scenario"]["planning_horizons"][0]),
        output:
            industrial_energy_demand_per_node="resources/demand/industrial_energy_demand_per_node_elec_s{simpl}_{clusters}_{planning_horizons}_{demand}.csv",
        threads: 1
        resources:
            mem_mb=1000,
        benchmark:
            "benchmarks/industrial_energy_demand_per_node_elec_s{simpl}_{clusters}_{planning_horizons}_{demand}.csv"
        script:
            "scripts/build_industry_demand.py"


if config["custom_data"].get("industry_demand", False) == False:

    rule build_industrial_distribution_key:  #default data
        input:
            regions_onshore=pypsaearth(
                "resources/bus_regions/regions_onshore_elec_s{simpl}_{clusters}.geojson"
            ),
            clustered_pop_layout="resources/population_shares/pop_layout_elec_s{simpl}_{clusters}.csv",
            industrial_database="data/industrial_database.csv",
            shapes_path=pypsaearth(
                "resources/bus_regions/regions_onshore_elec_s{simpl}_{clusters}.geojson"
            ),
        output:
            industrial_distribution_key="resources/demand/industrial_distribution_key_elec_s{simpl}_{clusters}.csv",
        threads: 1
        resources:
            mem_mb=1000,
        benchmark:
            "benchmarks/build_industrial_distribution_key_elec_s{simpl}_{clusters}"
        script:
            "scripts/build_industrial_distribution_key.py"

    rule build_industrial_production_per_country_tomorrow:  #default data
        input:
            industrial_production_per_country="data/industrial_production_per_country.csv",
        output:
            industrial_production_per_country_tomorrow="resources/demand/industrial_production_per_country_tomorrow_{planning_horizons}_{demand}.csv",
        threads: 1
        resources:
            mem_mb=1000,
        benchmark:
            "benchmarks/build_industrial_production_per_country_tomorrow_{planning_horizons}_{demand}"
        script:
            "scripts/build_industrial_production_tomorrow.py"

    rule build_industry_demand:  #default data
        input:
            industry_sector_ratios="data/industry_sector_ratios.csv",
            industrial_distribution_key="resources/demand/industrial_distribution_key_elec_s{simpl}_{clusters}.csv",
            industrial_production_per_country_tomorrow="resources/demand/industrial_production_per_country_tomorrow_{planning_horizons}_{demand}.csv",
            industrial_production_per_country="data/industrial_production_per_country.csv",
            costs=CDIR
            + "costs_{}.csv".format(config["scenario"]["planning_horizons"][0]),
        output:
            industrial_energy_demand_per_node="resources/demand/industrial_energy_demand_per_node_elec_s{simpl}_{clusters}_{planning_horizons}_{demand}.csv",
        threads: 1
        resources:
            mem_mb=1000,
        benchmark:
            "benchmarks/industrial_energy_demand_per_node_elec_s{simpl}_{clusters}_{planning_horizons}_{demand}.csv"
        script:
            "scripts/build_industry_demand.py"
