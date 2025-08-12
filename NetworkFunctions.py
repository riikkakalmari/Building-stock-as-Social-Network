# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 13:31:06 2025

@author: pvrika cite{}
"""


import sys
import json
#import urllib.request as url
import numpy as np
#import numpy.ma as ma
#import networkx as nx
import pandas as pd
#from random import sample
#import time
from scipy.cluster.hierarchy import dendrogram, to_tree, fcluster
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
import plotly.graph_objs as go
from BuildingNetwork import Buildingdata, Pregraph, Network, Dendrogram
import plotly.offline as po




def combinations(iterable, r):
    # combinations('ABCD', 2) → AB AC AD BC BD CD
    # combinations(range(4), 3) → 012 013 023 123

    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))

    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)
        

def dist(set1: set, set2: set):
    """
    Jaccard distance between two sets is one minus the ratio between the cardinality of the intersection and the union of the sets. 

    Parameters
    ----------
    set1 : set
        A set of objects.
    set2 : set
        A set of objects.

    Returns
    -------
    float:
        Jaccard distance between set1 and set2.

    """
    
    try:
        pleh = len(set.union(set1, set2))
            
        apu = len(set.intersection(set1, set2))
    
        return 1-apu/pleh
    
    except:
        if set1 == None or set2 == None:
            return None

def distance(df, dg):
        
        zips = zip(map(lambda key: df.to_dict('index')[key], df.to_dict('index').keys()),
              map(lambda key: dg.to_dict('index')[key], dg.to_dict('index').keys()))
    
        return pd.Series(dict(zip(df.index.values, list(map(lambda x: dist(set(x[0].items()), set(x[1].items())), zips)))))


class Flow(Network):
    def __init__(self, starttime: int, timestep: int):
        
        """
        The building network is a complete graph with the buildings as the nodes. Flow combines the the networks to time flow.

        Parameters
        ----------
        files : list of strings
            The paths and filenames of the building stock datas to be added to flow.

        Returns
        -------
        A class with the class Network as a parent class and an added attribute self.flow.
        """
        self.props = {}
        self.__start = starttime
        self.__time = starttime
        self.__step = timestep
        self.__buildingdatas = None
        self.__networks = None
        self.__flowframe = None

    @property
    def start(self):
        return self.__start

    @property
    def time(self):
        return self.__time
    
    @property
    def step(self):
        return self.__step

    @property
    def buildingdatas(self):
            return self.__buildingdatas
        
    @buildingdatas.setter
    def buildingdatas(self, files):
            data0 = Buildingdata(files[0])
            self.props[self.__time] = data0.properties
            multiindex = pd.MultiIndex.from_arrays([self.__start*np.ones((data0.data.shape[0])), np.array(data0.data.index.values)])
            datas = data0.data.set_index(multiindex)
            
            if files[0][:1] == 'bd':
                for file in files[1:]:
                
                    data = Buildingdata(file)
                    self.__time += self.__step
                    self.props[self.__time] = data.properties
                    
                    multiindex = pd.MultiIndex.from_arrays([self.__time*np.ones(data.data.shape[0]), np.array(data.data.index.values)])
                    data.data = data.data.set_index(multiindex)
                
                    datas = pd.concat([datas, data.data])
                
            else:
                for file in files[1:]:
                    breakswitch = 0
                
                    data = Buildingdata(file)
                    self.__time += self.__step
                    self.props[self.__time] = data.properties
                
                
                    for name in data0.idefix:
                        if breakswitch == 1:
                            break
                        for value in data0.data.loc[:, name].values:
                            if breakswitch == 1:
                                break
                            for column in data.idefix:
                                if value in data.data.loc[:, column].values:
                                    df = data0.data.loc[:, data0.idefix].merge(data.data, left_on = name, right_on = column, how = 'outer', suffixes = ['_x', ''])
                                    df = df.loc[:, data.data.columns]
                
                                    multiindex = pd.MultiIndex.from_arrays([self.__time*np.ones(data.data.shape[0]), np.array(df.index.values)])
                                
                                    df = df.set_index(multiindex)
                                
                                    datas = pd.concat([datas, df])
                                
                                    breakswitch = 1
                                    data0 = data
                                    break
        
            self.__buildingdatas = datas
        
    @property
    def networks(self):
            
            return self.__networks
        
    @networks.setter
    def networks(self, files):
            nets = []
            arr = np.arange(self.__start, self.__time + self.__step, self.__step)
            for t in arr:
                net = Network(self.buildingdatas.loc[t, :].dropna(how = 'all').dropna(how = 'all', axis = 1))
            
                nets.append(net)
                
            self.__networks = nets
        
    @property
    def flowframe(self):            
            
            return self.__flowframe
        
    @flowframe.setter
    def flowframe(self, files):
        try:
            self.__flowframe = pd.read_csv('flowframe.csv')
        except:
            if len(self.props.keys()) > 2:
                
                self.__flowframe = pd.concat(map(lambda t: distance(self.buildingdatas.loc[t, :][self.props[t]], self.buildingdatas.loc[t+1, :][self.props[t+self.__step]]), np.arange(self.__start, self.__time - self.__step, self.__step)))
    
            else:
                self.__flowframe = distance(self.buildingdatas.loc[self.__start, :][self.props[self.__start]], self.buildingdatas.loc[self.__time, :][self.props[self.__time]])

class Sankey(Dendrogram):
    '''
    
    Parameters
    ----------
    
    
    Returns
    -------
    
    
    '''
    
    def __init__(self, files: list, clustersize: int, clusternumber: int):
        self.__clustersize = clustersize
        self.__clusternumber = clusternumber
        self.dendrograms = files
        self.linkframe = files
        self.nodeframe = files
        
    @property
    def dendrograms(self):
        return self.__dendrograms

    @dendrograms.setter
    def dendrograms(self, files):
        L = []
        for file in files:
            L.append(Dendrogram(file, self.__clustersize, self.__clusternumber))
        self.__dendrograms = L    
        
    @property
    def linkframe(self):
        return self.__linkframe

    @linkframe.setter            
    def linkframe(self, files):
        try:
            self.__linkframe = pd.read_csv(f"linkframe['{self.__clustersize}', '{self.__clusternumber}'].csv", index_col=0)
        
        except:
        
            df = pd.DataFrame(np.concatenate([self.__dendrograms[0].clusterings, self.__dendrograms[1].clusterings]).reshape(2, self.__dendrograms[0].clusterings.shape[0]).T)
                
            df.columns = ['Source', 'Target']
                
            D = {'source': [], 'target': [], 'value': []}
            for key, group in df.groupby('Source'):
                if len(group) > self.__clustersize:
                    for avain, joukko in group.groupby('Target'):
                        D['source'].append(key)
                        D['target'].append(avain) 
                        D['value'].append(len(joukko))
                    
                    
            D = pd.DataFrame(D)
            D.loc[:, 'target'] = D['target'] + np.max(D['source'])
            
            
                
            self.__linkframe = D
        
    @property
    def nodeframe(self):
        return self.__nodeframe
    
    @nodeframe.setter
    def nodeframe(self, files):
        try:
            self.__nodeframe = pd.read_csv(f"nodeframe['{self.__clustersize}', '{self.__clusternumber}'].csv", index_col=0)
        
        except:
            df = pd.DataFrame()
                
            df['ID'] = np.concatenate([np.unique(self.__linkframe['source']), np.unique(self.__linkframe['target'])])
            
            df['label']=np.concatenate([[str(self.__dendrograms[0].clusterdata.index[np.argmax(self.__dendrograms[0].clusterdata[i])]) for i in np.unique(self.__linkframe['source'])], [str(self.__dendrograms[1].clusterdata.index[np.argmax(self.__dendrograms[1].clusterdata[i-np.max(self.__linkframe['source'])])]) for i in np.unique(self.__linkframe['target'])]])
    
            df = df.merge(pd.read_csv('colors.csv', index_col=0), how='inner')
    
            self.__nodeframe = df
            
        
       
    def fplot(self, fig: str):
        '''
        Sankey plot setup
        '''
        
        self.__linkframe['color'] = np.zeros(self.__linkframe.shape[0])
        
        for i in range(np.unique(self.__linkframe['source']).shape[0]):
            self.__linkframe.loc[self.__linkframe['source'] == self.__nodeframe.loc[i, 'ID'], 'color'] = self.__nodeframe.loc[i, 'color']
        
        data_trace = dict(
        type='sankey',
        domain = dict(
          x =  [0,1],
          y =  [0,1]
        ),
        orientation = "h",
        valueformat = ".0f",
        node = dict(
          pad = 10,
        # thickness = 30,
          line = dict(
            color = "black",
            width = 0
          ),
          label =  self.__nodeframe['label'].dropna(axis=0, how='any'),
          color = self.__nodeframe['color']
        ),
        link = dict(
          source = self.__linkframe['source'].dropna(axis=0, how='any'),
          target = self.__linkframe['target'].dropna(axis=0, how='any'),
          value = self.__linkframe['value'].dropna(axis=0, how='any'),
          color = self.__linkframe['color'].dropna(axis=0, how='any'),
          )
        )
    
        layout = dict(
            title = f"Sankey Diagram with min. clustersize: {self.__clustersize}",
        height = 772,
        font = dict(
          size = 10),)
    
        fig = dict(data=[data_trace], layout=layout)
        po.plot(fig, validate=False)
            
     

def main():
    
    flow = Flow(int(sys.argv[1]), int(sys.argv[2]))
    '''   
    flow.buildingdatas = sys.argv[5:]
    
    files = []
    
    i = 0
    for t in range(flow.start, flow.time+flow.step, flow.step):
        tallennus = 'bd_' + sys.argv[5:][i]
        
        flow.buildingdatas.loc[t].to_csv(tallennus)
        files.append(tallennus)
        
        i += 1
    '''
    files = sys.argv[5:]
    flow.buildingdatas = files
        
    flow.flowframe = files
        
    flow.flowframe.to_csv('flowframe.csv')
        
    flow.networks = files
        
    i = 0
    for file in files:
        tallennus = f'distanceframe_{file}'
        np.savetxt(tallennus, flow.networks[i].distanceframe, delimiter=',')
            
        tallennus = f'nodelista_{file}'
            
        flow.networks[i].network.to_csv(tallennus)
                
        tallennus = f'Z_{file}'
            
        np.savetxt(tallennus, flow.networks[i].Z, delimiter=',')
            
        i += 1

    sankey = Sankey(files, int(sys.argv[3]), int(sys.argv[4]))

    tallennus = f'linkframe{sys.argv[3:5]}.csv'
    sankey.linkframe.to_csv(tallennus)
    
    tallennus = f'nodeframe{sys.argv[3:5]}.csv'
    sankey.nodeframe.to_csv(tallennus)
    
    tallennus = 'Sankey{sys.argv[3:5]}'
    sankey.fplot()
    #plt.savefig(tallennus, format="pdf")
    
    
if __name__ == '__main__':
    main()