#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --error='job-%j-error.out'
#SBATCH --output='job-%j-out.out'
#SBATCH --export=ALL
#SBATCH --chdir= PATH_TO_PES

module purge
module load Anaconda3
module load Java
source activate #PATH_TO_ENV

export GRB_LICENSE_FILE= #PATH_TO_SOLVER_LIC

#rm PATH_TO_ELEC_NC
#rm PATH_TO_ELEC_NC

cp config.pypsa-earth_conservative_2030.yaml config.pypsa-earth.yaml
cp config_2030_cons.yaml config.yaml

snakemake -j 32 plot_summary

cp config_2030_cons_Qs.yaml config.yaml

snakemake -j 32 plot_summary


#rm PATH_TO_ELEC_NC
#rm PATH_TO_ELEC_NC

cp config.pypsa-earth_realistic_2030.yaml config.pypsa-earth.yaml
cp config_2030_real.yaml config.yaml

snakemake -j 32 plot_summary

cp config_2030_real_Qs.yaml config.yaml

snakemake -j 32 plot_summary

#rm PATH_TO_ELEC_NC
#rm PATH_TO_ELEC_NC

cp config.pypsa-earth_optimistic_2030.yaml config.pypsa-earth.yaml
cp config_2030_opt.yaml config.yaml

snakemake -j 32 plot_summary

cp config_2030_opt_Qs.yaml config.yaml

snakemake -j 32 plot_summary
