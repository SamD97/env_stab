 #!/bin/bash
echo ' Driver Script:'
echo ' 1. generates data in parallel'
echo ' 2. compresses the data files'
echo ' 3. analyzes/plots the data'

# data generation
parallel ::: 'python data_generator.py 1000' 'python data_generator.py 1000'
wait

# data compression
echo ' Compressing ...'
tar -cvzf data.tar.gz *.npy --remove-files
echo ' Compression Done!'

# data analysis
python data_plotter.py

echo ' Driver Done!'