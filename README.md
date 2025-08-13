Description
===========

Building-stock-as-Social-Network is an automated data processing routine that converts time dependent building stock data to a network structure between invidual buildings. The graph composition allows the analysis of the building stock and its' developement trough time using the methods of the social network analysis. The aim is to keep the program as simple and light weight as possible so that it is easily adaptable for user's needs. It is based on python packages Numpy, Pandas, Scikit-learn and Scipy.hierarchical.clustering. The program is divided into two modules, BuildingNetwork and StocksNetwork. Former handles building stock's status at observation time and later arranges the networks to the time axis. 

BuildingNetwork takes a filename or filepath as a variable and returns a complete graph with invidual buildings as the nodes. The edge lenghts are the Jaccard distances between buildings. Module consists of five classes. First class reads given datafile to a pandas.DataFrame with the information of each building as a row. It also difrentiates building features to identificating and shared properties. Identificating properties are unique for each building shuch as registeration number or coordinates. Shared include features like function or building material. Second class Pregraph forms network with two types of nodes. Building nodes are connected to property nodes as per their features but not to each other. The distances between the buildings is then computed using the fact that neighboring nodes are all shared property values. The Network class gives the linkage matrix used by scipy.hierarchical.clustering and the list of nodes. The Dendrgram class produces the clustering according to desired minimum cluster size and maximum number of clusters. BuildingNetwork can be used on its own if there is only one observation time or the data sets are large. If used as main it will save all class attributes to be set from memory from then on.


Prerequisites
=============

The module is based on the following packages:
    * NumPy
    * Pandas
    * Scikit-learn
    * scipy.hierarchical.clustering

.. :bibliography::

Installation
============
Anaconda Distribution comes with NumPy, Pandas and Scikit-learn -packages. 
Pandas and NumPy are easy to install with pip but Scikit-learn needs more care. It is why we recommend using Anaconda Navigator.   
After this simple copy-paste of the files BuildingNetwork.py, StocksNetwork.py and NetworkFuctions.py is enough.

Contributing
============
This work is funded by Maj and Tor Nessling Foundation.

Licence
=======
This work is licensed under CC BY-NC-SA 4.0:

https://creativecommons.org/licenses/by-nc-sa/4.0/


Citation
========


Contact
=======
riikka.kalmari@protonmail.com
