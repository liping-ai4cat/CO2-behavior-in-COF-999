#!/usr/bin/env python

import os
import shutil
import subprocess
import time

def get_job_count(user):
    """Get the number of jobs for the specified user."""
    result = subprocess.run(["squeue -u " + user], shell=True, capture_output=True, text=True)
    return len(result.stdout.splitlines())

restart = """
# File size threshold in bytes (200 MB)
threshold=$((200 * 1024 * 1024))

# Check if both CHGCAR and WAVECAR exist and are larger than 100 MB
if [[ -f "CHGCAR" && -f "WAVECAR" ]]; then
    size_chgcar=$(stat -c %s "CHGCAR")
    size_wavecar=$(stat -c %s "WAVECAR")

    if [[ $size_chgcar -gt $threshold && $size_wavecar -gt $threshold ]]; then
        echo "Both CHGCAR and WAVECAR are larger than 100 MB, skipping mpirun."
    else
        echo "Running VASP as not both CHGCAR and WAVECAR are larger than 100 MB."
        mpirun -np $MACHINE_COUNT vasp_std
    fi
else
    echo "One or both files CHGCAR and WAVECAR do not exist."
    mpirun -np $MACHINE_COUNT vasp_std
fi

current_dir=$(basename "$(pwd)")
new_dir="../${current_dir}_restart"

# Check if restart directory does not exist and create it
if [[ ! -d $new_dir ]]; then
    mkdir -p $new_dir
    echo "Directory $new_dir created."
fi

# Move to the new directory and execute restart script
cd $new_dir
cp ../Script_restart.py ./
./Script_restart.py ${current_dir}

# Running VASP in the new directory
mpirun -np $MACHINE_COUNT vasp_std

# Clean up large files after computation
rm CHGCAR WAVECAR
# Optional: Clean up in the original directory if needed
rm ../${current_dir}/CHGCAR ../${current_dir}/WAVECAR
"""

run = 'mpirun -np $MACHINE_COUNT vasp_std' 

usr = """
lipingliu"""
home_dir = os.getcwd()

job_limit = 50
sleep_time = 10

from ase import io
import numpy as np
import random

selected = ['train', 'val']
#selected = ['val']

mag_settings = ["mid"]

script_folder = './computation'

print(f"Structures to process: {selected}")

for n in selected:

    while get_job_count(usr) > job_limit:
        print(f'Hit the limit of jobs ({job_limit}). Sleeping for {sleep_time} minutes...')
        time.sleep(sleep_time * 60)

    n = str(n)
    m = str(n)

    for mag in mag_settings:
        job_name = f"{m}"
        jobfolder = job_name
        print(jobfolder)

        if not os.path.exists(jobfolder):
            os.mkdir(jobfolder)
            script_name = "Script.py"
            subfile_name = "nequip_falcon.qsub"
            os.system(f'cp {script_folder}/* {jobfolder}')
            os.chdir(jobfolder)

            os.system(f"sed -i 's/MMM/{n}/g' {script_name}")
            os.system(f"sed -i 's/MMM/{n}/g' {subfile_name}")

            # Run the script
            #subprocess.run(f'python {script_name}', shell=True)

            # Submit the job
            # os.system(f"echo '{run}' >> INCAR.qsub")
            os.system("sbatch nequip_falcon.qsub")

            os.chdir(home_dir)
