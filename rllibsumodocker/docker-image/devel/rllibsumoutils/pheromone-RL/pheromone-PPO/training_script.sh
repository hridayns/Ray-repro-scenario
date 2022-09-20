#!/bin/bash

set -e

# SUMO-dev environmental vars
export SUMO_HOME="/home/alice/sumo"
export SUMO_BIN="$SUMO_HOME/bin"
export SUMO_TOOLS="$SUMO_HOME/tools"
export PATH="$SUMO_BIN:$PATH"

fast=false
mixed_blockage=false
nopolicy=false

for arg in "$@"
do
    case $arg in
        -nv=*|--num_veh=*)
        nv="${arg#*=}"
        ;;
        -nl=*|--num_lanes=*)
        nl="${arg#*=}"
        ;;
        -nz=*|--num_zones=*)
        nz="${arg#*=}"
        ;;
        -ls=*|--lane_size=*)
        ls="${arg#*=}"
        ;;
        -ec=*|--evaporation=*)
        ec="${arg#*=}"
        ;;
        -df=*|--diffusion=*)
        df="${arg#*=}"
        ;;
        -nb=*|--num_blockages=*)
        nb="${arg#*=}"
        ;;
        -bls=*|--block_sizes=*)
        IFS=';' read -ra bls <<< "${arg#*=}"
        ;;
        -blp=*|--block_pos=*)
        IFS=';' read -ra blp <<< "${arg#*=}"
        ;;
        -bll=*|--block_lanes=*)
        IFS=';' read -ra bll <<< "${arg#*=}"
        ;;
        -bdur=*|--block_duration=*)
        IFS=';' read -ra bdur <<< "${arg#*=}"
        ;;
        -puf=*|--pf_update_freq=*)
        puf="${arg#*=}"
        ;;
        # additional options for train.py / run.py
        -mbt=*|--mixed_blockage_training=*)
        mixed_blockage="${arg#*=}"
        ;;
        -f=*|--fast=*)
        fast="${arg#*=}"
        ;;
        -nopol=*|--nopolicy=*)
        nopolicy="${arg#*=}"
        ;;
        *)
        # unknown option
        ;;
    esac
done

mkdir -p ./data

if [[ "$nopolicy" = true ]]; then
    python modifyScenario.py -nv $nv -nl $nl -ls $ls -nb $nb -bls $bls -blp $blp -bll $bll -bdur $bdur -puf $puf --nopolicy
else
    if [[ "$mixed_blockage" = true ]]; then
        python modifyScenario.py -nv $nv -nl $nl -nz $nz -ls $ls -ec $ec -df $df -nb $nb -bls $bls -blp $blp -bll $bll -bdur $bdur -puf $puf --mixed_blockage_training
    else
        python modifyScenario.py -nv $nv -nl $nl -nz $nz -ls $ls -ec $ec -df $df -nb $nb -bls $bls -blp $blp -bll $bll -bdur $bdur -puf $puf
    fi
fi

cd scenario
bash ./gen_network.sh
cd ..

cp -f /home/alice/devel/rllibsumoutils/rllibsumoutils/sumoconnector.py /home/alice/libraries/rllibsumoutils/rllibsumoutils/sumoconnector.py

if [[ "$fast" = true ]]; then
    echo "NO GUI"
    if [[ "$nopolicy" = true ]]; then
        python train.py -nv $nv -nl $nl -ls $ls -nb $nb -bls $bls -blp $blp -bll $bll -bdur $bdur -puf $puf --fast --nopolicy
    else
        if [[ "$mixed_blockage" = true ]]; then
            python train.py -nv $nv -nl $nl -nz $nz -ls $ls -ec $ec -df $df -nb $nb -bls $bls -blp $blp -bll $bll -bdur $bdur -puf $puf --fast --mixed_blockage_training
        else
            python train.py -nv $nv -nl $nl -nz $nz -ls $ls -ec $ec -df $df -nb $nb -bls $bls -blp $blp -bll $bll -bdur $bdur -puf $puf --fast
        fi
        
    fi
else
    echo "GUI"
    if [[ "$nopolicy" = true ]]; then
        python train.py -nv $nv -nl $nl -ls $ls -nb $nb -bls $bls -blp $blp -bll $bll -bdur $bdur -puf $puf --nopolicy
    else
        if [[ "$mixed_blockage" = true ]]; then
            python train.py -nv $nv -nl $nl -nz $nz -ls $ls -ec $ec -df $df -nb $nb -bls $bls -blp $blp -bll $bll -bdur $bdur -puf $puf --fast --mixed_blockage_training
        else
            python train.py -nv $nv -nl $nl -nz $nz -ls $ls -ec $ec -df $df -nb $nb -bls $bls -blp $blp -bll $bll -bdur $bdur -puf $puf --fast
        fi
    fi
    # python train.py -nv $nv -nl $nl -nz $nz -ls $ls -ec $ec -df $df -nb $nb -bls $bls -blp $blp -bll $bll
fi


# if [[ "$nopolicy" = false ]]; then
#     bash ./plot.sh
# fi

# ./training_script.sh -nv=100 -nl=2 -nz=4 -ls=1000 -ec=0.3 -df=0.5 -nb=1 -bls=250 -blp=750 -bll=0 -bdur='END' -puf=1 --fast=true --nopolicy=false

# ./training_script.sh -nv=1000 -nl=3 -nz=10 -ls=5000 -ec=0.3 -df=0.5 -nb=1 -bls=500 -blp=2750 -bll=0 -bdur='END' -puf=1 --fast=true --nopolicy=false -mbt