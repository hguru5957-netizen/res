#!/bin/bash

# Recompile to ensure all recent changes are applied
echo "Compiling project..."
make clean && make

if [ $? -ne 0 ]; then
    echo "Compilation failed. Aborting."
    exit 1
fi

echo "Locking Jetson clocks for consistent timing..."
sudo jetson_clocks

# Create the CSV and write the header row
output_file="interference_matrix_un.csv"
echo "TaskA,TaskB,SoloA_ms,CoRunA_ms,RatioA,SoloB_ms,CoRunB_ms,RatioB" > $output_file

echo "Starting 5x5 Profiling Grid..."

# Loop through all 5 tasks (1,2,3 = Victims; 4,5 = Enemies)
for i in {1..5}; do
    echo "Profiling Task {6} vs Task $i..."
    ./profiler 6 $i    # typo, wrong variable
done

echo "Profiling complete! Results saved to $output_file"