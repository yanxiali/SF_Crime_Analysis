# SF_Crime_Analysis

**Results**: 
Please check out the jupyter notebook of the project [here](https://github.com/yanxiali/SF_Crime_Analysis/blob/master/SF_Crime.ipynb). If you experience loading problems (as it is a big file), please take a look of a markdown copy of the project [here](https://github.com/yanxiali/SF_Crime_Analysis/blob/master/results/SF_Crime.md)

**Keywords**:
time series analysis - ARIMA model - visualization 

**Description**:
In this project, I would like to know the nicest neighborhoods in San Francisco that have the lowest crime rates, in case I decide to move there after I graduate from Hawaii. Therefore, I retrieved the 15-year data (between 2003 and 2018) of crime events, from the San Francesco Police Department (https://data.sfgov.org/Public-Safety/-Change-Notice-Police-Department-Incidents/tmnf-yvry).

During EDA, I found some interesting results. For instance, among all types of crime (e.g., “Theft”, “Assault”, “Drug”), “Theft” is the most frequent one (22%). Also, midnight and noon are the periods with more crime per day and Friday is the most dangerous day per week. January and March both have the most number of the incidents, as opposed to the holiday seasons (in December and January). 

Furthermore, I explored and visualized the spatial distribution of the incidents on SF maps. I also analyzed how they interact with each other on heat maps. I also performed time series analysis to check how different crimes have evolved with time. I then fitted the data with an ARIMA model, combined with hyper-parameter tuning, to forecast the crime rates in the next year. 

A future follow-up of this project could be investigating the house price in SF, which is another very important factor for my moving plans. 





