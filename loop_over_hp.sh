#!/bin/bash

# Base path to the 'hp' folder
base_path="../data/testing/"

# Loop through all folders inside the 'hp' directory
for folder in "$base_path"/hp_*; do
    echo " "
	echo "-----------------------------------"
    echo "     /\_/\      /\_/\      /\_/\\ "
    echo "    ( ^_^ )    ( o.o )    ( ^.^ ) "
    echo "     > ^ <      > ^ <     (~) (~) "
    echo "-----------------------------------"
    if [[ -d "$folder" ]]; then  # Ensure it's a directory
        folder_name=$(basename "$folder")  # Get just the folder name (e.g., hp_333)
        # Extract the number after "hp_"
        hp_number=$(echo "$folder_name" | grep -o 'hp_[0-9]*' | cut -d'_' -f2)
        echo "Folder: $folder_name"

        # Run your Python script and pass the HP number
        # Uncomment and adjust the line below if needed
        #conda activate gbc
        python build_features_and_make_predictions_multiple_hp.py "$hp_number"
    fi
done
