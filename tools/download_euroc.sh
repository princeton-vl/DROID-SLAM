#!/bin/bash
set -euo pipefail

# where to put everything
EUROC_PATH="datasets/EuRoC"

# list of all sequences
declare -a evalset=(
    MH_01_easy
    MH_02_easy
    MH_03_medium
    MH_04_difficult
    MH_05_difficult
    V1_01_easy
    V1_02_medium
    V1_03_difficult
    V2_01_easy
    V2_02_medium
    V2_03_difficult
)

# make sure base dir exists
mkdir -p "${EUROC_PATH}"

for scene in "${evalset[@]}"; do
    # full URL still needs the right folder on the server
    if [[ "${scene}" == MH* ]]; then
        url="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/${scene}/${scene}.zip"
    elif [[ "${scene}" == V1* ]]; then
        url="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/${scene}/${scene}.zip"
    else
        url="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/${scene}/${scene}.zip"
    fi

    # local paths
    zipfile="${EUROC_PATH}/${scene}.zip"
    outdir="${EUROC_PATH}/${scene}"

    mkdir -p "${outdir}"

    echo "Downloading ${scene}..."
    wget -c "${url}" -O "${zipfile}"

    echo " Unzipping into ${outdir}/..."
    unzip -o "${zipfile}" -d "${outdir}"

    echo " Cleaning up..."
    rm "${zipfile}"

    echo "✔ Done with ${scene}"
done
