#!/bin/bash

# URL of the file to download
FOLDER="data_sqa"
URL="https://download.microsoft.com/download/1/D/C/1DC270D2-1B53-4A61-A2E3-88AB3E4E6E1F/SQA%20Release%201.0.zip"

mkdir -p "${FOLDER}"
cd "${FOLDER}"
echo "Downloading SQA..."
curl -L -o "SQA Release 1.0.zip" "$URL"

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Download completed successfully!"
   
    # Unzip the file
    echo "Extracting files..."
    unzip "SQA Release 1.0.zip"
   
    # Check if unzip was successful
    if [ $? -eq 0 ]; then
        echo "Extraction completed successfully!"
        
        # Move files from the SQA Release directory to current directory, preserving table_csv
        echo "Moving files and cleaning up..."
        mv "SQA Release 1.0"/* .
        rmdir "SQA Release 1.0"
        rm "SQA Release 1.0.zip"
        
        echo "Files can be found in the ${FOLDER} directory"
    else
        echo "Error: Failed to extract the zip file."
        exit 1
    fi
else
    echo "Error: Failed to download the file."
    exit 1
fi