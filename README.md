### Documentation for lockdown_period_analysis.py

#### Overview
This script processes weather data files from multiple cities, performs statistical analysis to determine if there were significant changes in weather patterns during the lockdown period of 2020, and generates visualizations to display the results. The analysis includes ANOVA and Tukey's HSD post-hoc tests for various weather parameters.

1. **Place your CSV files** in a directory. Each file should be named after the city it represents (e.g., `Montreal.csv`).

2. **Run the script** from the command line, specifying the directory containing the CSV files:
python3 main.py /path/to/your/data/directory


**Running testing.py**:

Ensure the CSV files for Halifax, Montreal, and Vancouver (split into two parts) are available in your specified directory.
Modify Code for Desired City:

Open the Python script.
Navigate to the section labeled #Read file section
