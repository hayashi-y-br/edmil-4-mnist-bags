#!/bin/sh
#$ -cwd
#$ -l gpu_h=1
#$ -l h_rt=24:00:00
#$ -t 1-5

cd /home/6/uf02776/edmil-4-mnist-bags
source .venv/bin/activate
python main.py -m model=AddMIL settings.lr=0.001,0.0005,0.0001,0.00005,0.00001 settings.wd=0,0.001,0.0001,0.00001 seed=$(($SGE_TASK_ID-1))