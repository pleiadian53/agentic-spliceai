#!/bin/bash

source constants.py

CLr=$((CL_max/2))
CLl=$(($CLr+1))
# First nucleotide not included by BEDtools

cat $splice_table | awk -v CLl=$CLl -v CLr=$CLr '{print $3"\t"($5-CLl)"\t"($6+CLr)}' > temp.bed
# cat $splice_table | awk -v CLl=$CLl -v CLr=$CLr '{print $1"\t"($4-CLl)"\t"($5+CLr)}' > temp.bed

# error checking
awk '$2 >= 0 && $3 > $2' temp.bed > filtered.bed

bedtools getfasta -bed filtered.bed -fi $ref_genome -fo $sequence -tab

# rm temp.bed filtered.bed
