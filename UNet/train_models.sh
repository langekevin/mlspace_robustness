#!/bin/bash

# Loop three times to create three new models
for ((i = 0; i < 3; i++)); do
    python main.py
done

echo "[I] Finished training the three models"
echo "[I] You will find the results in the folder ../models"
