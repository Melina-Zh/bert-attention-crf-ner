#!/bin/bash
#SBATCH -o ./result/BERT/%j.log
#SBATCH -J BERT
#SBATCH --gres=gpu:V100:1 
#SBATCH -c 5 
job_name="BERT"
date_time=`date +%Y%m%d-%H%M%S`
mkdir ./result/${job_name}/${date_time}
filename="./result/"${job_name}'/'${date_time}"/five_res.log"
echo ${filename}
acc_std_name="./result/"${job_name}'/'${date_time}"/acc_std.log"
source /home/LAB/anaconda3/etc/profile.d/conda.sh
conda activate bert-torch-36
echo "./result/${job_name}/${date_time}/"
for i in {1..5};do
	python main.py train --checkpoint="./result/${job_name}/${date_time}/"
done
python cal.py ${filename} ${acc_std_name}
mv ./result/${job_name}/$SLURM_JOB_ID.log ./result/${job_name}/${date_time}/${job_name}.log
