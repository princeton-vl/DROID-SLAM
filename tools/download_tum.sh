#!/bin/bash
set -euo pipefail

# where to put everything
TUM_PATH="datasets/TUM-RGBD"

# list of all sequences
declare -a evalset=(
    rgbd_dataset_freiburg1_360
    rgbd_dataset_freiburg1_desk
    rgbd_dataset_freiburg1_desk2
    rgbd_dataset_freiburg1_floor
    rgbd_dataset_freiburg1_plant
    rgbd_dataset_freiburg1_room
    rgbd_dataset_freiburg1_rpy
    rgbd_dataset_freiburg1_teddy
    rgbd_dataset_freiburg1_xyz
)

# make sure base dir exists
mkdir -p "${TUM_PATH}"

for scene in "${evalset[@]}"; do
    # full URL still needs the right folder on the server
    url="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/${scene}.tgz"

    # local paths
    tarfile="${TUM_PATH}/${scene}.tgz"
    outdir="${TUM_PATH}"

    mkdir -p "${outdir}"

    echo "Downloading ${scene}..."
    wget -c "${url}" -O "${tarfile}"

    echo " Unzipping into ${outdir}/..."
    tar -zxvf "${tarfile}" -C "${outdir}"

    echo " Cleaning up..."
    rm "${tarfile}"

    echo "✔ Done with ${scene}"
done
