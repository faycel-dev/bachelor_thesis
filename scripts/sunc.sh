run_rsync() {
  echo started syncing to node "$1"
  srun -p csedu-prio -w "$1" --qos csedu-small mkdir -p /scratch/"$USER"
  srun -p csedu-prio -w "$1" --qos csedu-small rsync cn84:/scratch/"$USER"/virtual_environments/ /scratch/"$USER"/virtual_environments/ -ah --delete
  echo completed syncing to node "$1"
}

if [[ "$HOSTNAME" != "cn84"* ]]; then
  echo run this script from cn84
  exit
fi

# gpu nodes
run_rsync cn47
