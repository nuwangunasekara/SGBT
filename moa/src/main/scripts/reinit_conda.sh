#!/bin/bash

conda_env_path=$1
conda_yml_file=$2

if [ $# -lt 2 ]; then
  echo "$#"
  echo "$0 <conda_env_path> <conda_yml_file>"
  exit 1
else
  echo  "$@"
fi


eval "$(conda shell.bash hook)"
conda init bash


re_init_conda_env ()
{
  echo "Removing conda env $conda_env_path"
  conda remove --prefix $conda_env_path --all -y

  echo "Creating conda env $conda_env_path from config $conda_yml_file"
  conda env create --prefix $conda_env_path --file $conda_yml_file

  echo "Updating conda env $conda_env_path from config $conda_yml_file"
  conda env update --prefix $conda_env_path --file $conda_yml_file  --prune

  conda activate $conda_env_path
  conda env list
}

re_init_conda_env
