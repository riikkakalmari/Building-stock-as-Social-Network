Description
===========

Building Network is an automated data processing routine that converts building stock data to form a network structure between invidual buildings.
The network structure is a complete graph with invidual buildings as the nodes. The edge lenghts are the Jaccard distances between buildings. 
The graph composition allows the analysis of the building stock and its' developement trough time using the methods of the social network analysis.

The aim is to keep the module as simple and light weight as possible so that it is easily adaptable for user's needs. 
Module consists of four classes. First class reads given datafile to a pandas.DataFrame with the information of each building as a row.
It also difrentiates building features to identificating and shared properties. Identificating properties are unique for each building shuch as registeration number or coordinates. Shared include features like function or building material.
Second class Pregraph forms network with two types of nodes. Building nodes are connected to property nodes as per their features but not to each other. The distances between the buildings is then computed using the fact that neighboring nodes are all shared property values.
The Network class  


Prerequisites
=============

The module is based on the following packages:
    * NumPy
    * Pandas
    * Scikit-learn AgglomerativeClustering

.. :bibliography::

Installation
============
Anaconda Distribution comes with NumPy, Pandas and Scikit-learn -packages. 
Pandas and NumPy are easy to install with pip but Scikit-learn needs more care. It is why we recommend using Anaconda Navigator.   
After this simple copy-paste of the files BuildingNetwork.py and NetworkFuctions.py is enough.

Contributing
============

Licence
=======
This work is licensed under CC BY-NC-SA 4.0:

https://creativecommons.org/licenses/by-nc-sa/4.0/


Citation
========


Contact
=======
riikka.kalmari@protonmail.com
