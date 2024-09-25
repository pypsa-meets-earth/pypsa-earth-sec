#!/bin/bash

micromamba activate pypsa-earth

cp config.bright_BI.yaml config.yaml
snakemake --profile slurm all

cp config.bright_DE.yaml config.yaml
snakemake --profile slurm all

cp config.bright_GH.yaml config.yaml
snakemake --profile slurm all

NEXTCLOUD_URL="https://tubcloud.tu-berlin.de/remote.php/webdav/BRIGHT/results/"
USERNAME="cpschau"
PASSWORD=$(get_nextcloud_password)

# Upload the file to Nextcloud via WebDAV
tar -czf results_0925.tar.gz /results/092524_test/
curl -u "$USERNAME:$PASSWORD" -T "results_0925.tar.gz" "$NEXTCLOUD_URL"