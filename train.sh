#!/bin/bash
#SBATCH -o ./result/BERT_CRF/%j.log
#SBATCH -J BERT_CRF 
#SBATCH --gres=gpu:V100:1 
#SBATCH -c 5 
job_name="BERT_CRF"
date_time=`date +%Y%m%d-%H%M%S`
mkdir ./result/${job_name}/${date_time}
filename="./result/"${job_name}'/'${date_time}"/five_res.log"
echo ${filename}
acc_std_name="./result/"${job_name}'/'${date_time}"/acc_std.log"
source /home/LAB/anaconda3/etc/profile.d/conda.sh
conda activate bert-torch-36
echo "./result/${job_name}/${date_time}/"
python main.py train --checkpoint="./result/${job_name}/${date_time}/"
python cal.py ${filename} ${acc_std_name}
mv ./result/${job_name}/$SLURM_JOB_ID.log ./result/${job_name}/${date_time}/${job_name}.log
