#!/bin/bash

TARTANAIR_PATH="datasets/tartanair_test"

mkdir -p "${TARTANAIR_PATH}"

gdown "1N8qoU-oEjRKdaKSrHPWA-xsnRtofR_jJ" --output "${TARTANAIR_PATH}/images.tar.gz"
wget -c "https://cmu.box.com/shared/static/3p1sf0eljfwrz4qgbpc6g95xtn2alyfk.zip" -O ${TARTANAIR_PATH}/groundtruth.zip

unzip -o ${TARTANAIR_PATH}/groundtruth.zip -d "${TARTANAIR_PATH}"
tar -zxvf "${TARTANAIR_PATH}/images.tar.gz" -C "${TARTANAIR_PATH}"


# rm ${TARTANAIR_PATH}/groundtruth.zip
# rm ${TARTANAIR_PATH}/images.zip
