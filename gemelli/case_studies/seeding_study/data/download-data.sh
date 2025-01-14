#!/bin/bash

sites=("Baby-Feces" "Baby-Forearm" "Baby-Mouth")
for site in "${sites[@]}"; do
    mkdir $site
    wget -O $site/metadata.tsv https://raw.githubusercontent.com/knightlab-analyses/seeding-study/master/data/split-data/$site/metadata.tsv
    wget -O $site/table.biom https://github.com/knightlab-analyses/seeding-study/raw/master/data/split-data/$site/table.biom
done

sites=("Baby-Feces-0-2" "Baby-Feces-7-360" "Baby-Forearm-0-2" "Baby-Forearm-7-360" "Baby-Mouth-0-2" "Baby-Mouth-7-360")
for site in "${sites[@]}"; do
    mkdir $site
    wget -O $site/metadata.qza https://raw.githubusercontent.com/knightlab-analyses/seeding-study/master/data/ctf-results/$site/metadata.qza
    wget -O $site/table.qza https://github.com/knightlab-analyses/seeding-study/raw/master/data/ctf-results/$site/table.qza
    # extract data to biom / tsv
    qiime tools export --input-path $site/table.qza --output-path $site/table-tmp
    mv $site/table-tmp/feature-table.biom $site/table.biom
    mv $site/metadata.qza $site/metadata.tsv
    rm -rf $site/table.qza
    rm -rf $site/table-tmp
done