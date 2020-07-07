# Graphical Association Analysis for Describing Variation in Surgical Providers
Code to extract MBS claims, create association rules, and rank providers by variation from commonly used claims within various procedures.<br/>

This repository contains an extract of a larger data analysis framework. The extract contains the code used in the paper submitted by the same name, available as a preprint at _____.

## Software version compatibility
This code was run in Python 3.6.8 in Ubuntu 18.04.2 LTS. Some graphs were made in R 3.4.4 using rpy2. Python package versions are documented in requirements.txt

## Structure
### Overview
The data analysis framework is split broadly into two sections:<br/>
• The 'utilities' folder contains tools used across data analyses, such as logging, graphing, and data extraction functions. It also contains an abstract base class and a template for data analyses<br/>
• The 'data analysis' folder (and other, removed folders) contain instances of various data analyses which conform to the base class, and additional tools associated with the collection of analyses.<br/>

These sections are linked together by run_analysis.py, which loads the data analysis file and calls the data extractions functions from within utilities/file_utils.py

### Usage
• Data location should be specified in utilities/file_utils.py<br/>
• Test parameters, including the data analysis file, should be specified in run_analysis.py<br/>
• The test can then be run with python run_analysis.py<br/>

### Data extraction summary
The base analysis class typically expects a general test order of: extract, combine, and process the data from the source, which is separated by years; process the data in some way to make it more suitable for the test; run the analysis. Loading previously extracted and processed data from a file is a possible alternative, which bypasses the extraction and processing steps.

Because the data source is separated by years, three ways of working with the data are implemented in run_analysis.py - combining the years together, iterating over the years outside the analysis, or iterating over the years within the analysis. Only the combined years are used in this repository.

### Analysis descriptions
#### regional_variation.py
This is one of the two main analyses. For each geographical region, an association rule-based graph model is created, and variations in the regions are compared.

#### provider_ranking.py
The other main analysis. A model for the nation and for each surgical provider is also created, and a modified graph edit distance is calculated to rank the providers by how much they vary from the national model.

#### basic_mba.py
Wrappers for association analysis functions, should not be run as an analysis.

#### dos_delta.py
Descriptive statistics for the difference in date of service for two items, nominally an anaesthetic and a surgical procedure

#### get_surgery_info.py
Claim counts for all items within a range or associated with another item (nominally for surgical procedures and anaethesia, respectively)

#### example_ged.py
Used to create the example figure in the paper

#### export_claims.py
Used to extract provider claims for validation

## Tests
Unit tests have been run within VSCode using the unittest framework.

## Contact
Please contact me with any questions
<a href=https://au.linkedin.com/in/james-kemp-11874a93><img src=https://blog-assets.hootsuite.com/wp-content/uploads/2018/09/In-2C-54px-R.png
    width = 18 height = 15 /></a>
<a href=https://www.researchgate.net/profile/James_Kemp6><img src=https://www.researchgate.net/apple-touch-icon-180x180.png
    width=15 height=15 /></a>