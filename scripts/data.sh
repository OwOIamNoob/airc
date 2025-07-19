#!/bin/bash
# Get files from Google Drive
fileID="1X2gVB5XDdQ1_krar80l3u2iSh9lcx3RC"
filename="dataset.zip"
# $2 = file name
gdown $fileID -O ./data/$filename
cd ./data
unzip $filename
cd ..