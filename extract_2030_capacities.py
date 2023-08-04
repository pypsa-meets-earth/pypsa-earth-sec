#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 21:40:26 2023

@author: haz43975
"""
import pypsa
import snakemake
import pandas as pd
import os

def extract_res(n):
        
    res_caps = pd.DataFrame(index=n.buses[n.buses.carrier=='AC'].index, columns=technologies)
        
    for tech in technologies:
        res_ind=n.generators[n.generators.carrier==tech].index
        res_caps_vals= n.generators.loc[res_ind, "p_nom_opt"]
        res_caps_vals.index = res_caps_vals.index.to_series().apply(lambda x: x.split(" ")[0])
        res_caps[tech] = res_caps_vals
    parent_dir = n_path.split("/")[0] + "/{}/optimal_capacities".format(run) 

    ex_q_30 = int(n_path.split('_')[-1].split('e')[0])
    ex_q_50 = ex_quantities_2050[ex_q_30]

    res_caps.to_csv(parent_dir+ "/" + n_path.split("/")[-1].\
                    replace(".nc", ".csv").replace("elec", "res_caps_elec").replace('_'+str(ex_q_30), '_'+str(ex_q_50)))

    
def extract_electrolyzers(n):
    
    parent_dir = n_path.split("/")[0] + "/{}/optimal_capacities".format(run)

    ex_q_30 = int(n_path.split('_')[-1].split('e')[0])
    ex_q_50 = ex_quantities_2050[ex_q_30]

    elec_ind =n.links[n.links.carrier=='H2 Electrolysis'].index
    n.links.loc[elec_ind].p_nom_opt.to_csv(parent_dir+ "/" +n_path.split("/")[-1]\
                    .replace(".nc", ".csv").replace("elec", "electrolyzer_caps_elec").replace('_'+str(ex_q_30), '_'+str(ex_q_50)))

def extract_pipelines(n):
    
    parent_dir = n_path.split("/")[0] + "/{}/optimal_capacities".format(run)

    ex_q_30 = int(n_path.split('_')[-1].split('e')[0])
    ex_q_50 = ex_quantities_2050[ex_q_30]

    ppl_ind =n.links[n.links.carrier=='H2 pipeline'].index
    n.links.loc[ppl_ind].p_nom_opt.to_csv(parent_dir+ "/" +n_path.split("/")[-1].\
                    replace(".nc", ".csv").replace("elec", "pipeline_caps_elec").replace('_'+str(ex_q_30), '_'+str(ex_q_50)))
    
    
if __name__ == "__main__":
    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    technologies= ['csp', 'rooftop-solar', 'solar', 'onwind', 'onwind2', 'offwind', 'offwind2']
    
    ex_quantities =   [0, 1, 10, 50, 100, 200, 1000]

    ex_quantities_2050 = {0:0, 1:10, 10:100, 50:500, 100:1000, 200:2000, 1000:3000}

    n_paths = []
    run = 'BR_2030_FINAL_flat_heat_2'
    clusters = {'NZ':147, 'AP':146, 'BS':145}
    i_rate = {'BS': {2030:0.175, 2050:0.175}, 'AP':{2030:0.076, 2050:0.086}, 'NZ':{2030:0.071, 2050:0.045}}
    ns = {}

    for year in [2030]:
        for s in ['BS', 'AP', 'NZ']:
                for export in ex_quantities:

                    n_paths.append("results/{}/postnetworks/elec_s_{}_ec_lc1.0_Co2L_3H_{}_{}_{}_{}export.nc".format(run, clusters[s], year, i_rate[s][year],s, export))
    
    for n_path in n_paths:
        n = pypsa.Network(n_path)
        clusters = n_path.split("_")[4]

        extract_res(n)
        extract_electrolyzers(n)
        extract_pipelines(n)