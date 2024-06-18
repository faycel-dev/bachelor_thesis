#! /usr/bin/env bash

# set variable to path where this script is
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

# make a directory on the ceph file system to store logs and checkpoints
# and make a symlink to access it directly from the root of the project
mkdir -p /scratch/"$USER"/lightning_logs
# ln -sfn /scratch/"$USER"/lightning_logs "$SCRIPT_DIR"/../lightning_logs

mkdir -p /scratch/"$USER"/logs
ln -sfn /scratch/"$USER"/logs "$SCRIPT_DIR"/../logs

# mkdir -p /scratch/"$USER"/whisper/data

echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN84 ###"
./setupvertual.sh

echo "### SETTING UP VIRTUAL ENVIRONMENT ON CN47 ###"
srun -p csedu-prio -A cseduproject -q csedu-small -w cn48 rsync -a cn84:/scratch/"$USER"/ /scratch/"$USER"/
srun -p csedu-prio -A cseduproject -q csedu-small -w cn47 rsync -a cn84:/scratch/"$USER"/ /scratch/"$USER"/

