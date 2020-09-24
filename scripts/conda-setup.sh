#!/usr/bin/env bash

# Define variables
name = "also_anomaly_detector"

fn_conda_dep="dependencies-deploy.yaml"
fn_conda_dev="dependencies-develop.yaml"


source ~/.bashrc
function conda_deploy() {
    source ~/.bashrc;
    cd root_dir;
    conda env create -f fn_conda_dep -n name
}

function conda_develop() {
    source ~/.bashrc;
    cd root_dir;
    conda activate name;
    conda env update -n name -f fn_conda_develop
}

function add_ipykernel() {
    source ~/.bashrc;
    cd root_dir;
    conda activate name;

    python -m ipykernel install --user --name name --display-name "$name"
}

conda_deploy;
conda_develop;
add_ipykernel;