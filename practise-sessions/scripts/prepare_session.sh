#!/bin/bash

for file in $(ls *.ipynb) 
do
  if [[ "${file}" =~ _a\.ipynb$ ]];
  then
    continue
  fi

  if [[ "${file}" =~ ^[0-9][0-9].*\.ipynb ]];
  then
    target_file=$(echo "$file" | sed "s/.ipynb/_a.ipynb/")
    cp $file $target_file
  fi
done
