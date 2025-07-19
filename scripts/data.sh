#!/bin/bash
# Get files from Google Drive
fileID="1X2gVB5XDdQ1_krar80l3u2iSh9lcx3RC"
filename="dataset.zip"
# $2 = file name
gdown $fileID -O ./outputs/$filename
cd ./outputs
unzip $filename
cd ..