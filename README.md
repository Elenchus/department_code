# Graphical Association Analysis for Describing Variation in Surgical Providers
Code to extract MBS claims, create association rules, and rank providers by variation from commonly used claims within various procedures

## Motivation
This repository contains an extract of a larger data analysis framework. This extract contains the code used in the paper submitted by the same name, available as a preprint at _____.

## Code style
PEP8 is vaguely adhered to...

## Dependencies
This code is confirmed to be working in Python 3.6.8.

## Structure
### Overview
The data analysis framework is split broadly into two sections:
• The 'utilities' folder contains tools used across data analyses, such as logging, graphing, and data extraction functions. It also contains an abstract base class and a template for data analyses<br/>
• The 'data analysis' folder (and other, removed folders) contain instances of various data analyses which conform to the base class.<br/>

These sections are linked together by run_analysis.py, which loads the data analysis file and calls the data extractions functions from within utilities/file_utils.py

### Usage
• Data location should be specified in utilities/file_utils.py<br/>
• Test parameters, including the data analysis file, should be specified in run_analysis.py<br/>
• The test can then be run with pyton run_analysis.py<br/>

## Tests
Unit tests have been run within VSCode using the unittest framework.

