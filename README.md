
# Relation between BMI and the estimation error in the number of steps

This respository contains the scripts that allow for analyses on a dataset collected by an activity tracker. 
The goal of the analysis is to determine whether a relation exists between the BMI and the error between the measured number of steps taken and the self reported number of steps.
Further explination can be found in the paper: Correlation between BMI and response bias: in the case of step counting by Pim de Ruijter.

## Getting Started

The instructions below give a step by step guide to get the code from this repository working locally.

### Prerequisites

The following software and packages are required for proper functioning. Later versions will likely also work.
- sklearn version 1.1
- scipy version 1.8.0
- python version 3.9
- pandas version 1.0.5
- matplotlib version 3.3.4

 ### Files
- **Data_IER_2022**
  - The dataset provided by the TU Delft.
  - It contains activity data on 286 students
- **BMI_Est_error.py**
  - The python script containing all the code used in this study.
  - It also includes the statistical analysis from the scipy library.


### Data specific information
- **The dataset is not to be shared, therefore explanations are only available to students of the TU Delft through the following link**
- https://brightspace.tudelft.nl/d2l/le/content/400888/viewContent/2422024/View

### Installing

- Clone the repository locally.
- Place the dataset in the same folder as the BMI_Est_error.py
- The dataset is unfortunatly not available for sharing.

## Running the tests

Open the python script and run it.

Genders can be filtered if desired by using the code on lines 99-107

### Sample Tests

In the script a t-test is performed to find the appropriate P-values.


## Authors

  - **Pim de Ruijter**
  - **Contact details:**
    - p.j.m.deruijter@student.tudelft.nl
