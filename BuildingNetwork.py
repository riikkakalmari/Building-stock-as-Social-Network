# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 18:50:19 2025

@author: pvrika
"""

import sys
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, to_tree, fcluster
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
from NetworkFunctions import combinations, dist, add_data 




# Data loading helper, currently for *.txt, *.csv, pandas.DataFrame.
class Buildingdata:

    def __init__(self, file: str):
        """
        A data organiser that will store the data. It also extracts the names of the columns that can be used as identification for the buildings. 

        Parameters
        ----------
        file : string
            The path and filename of the building stock data to be analysized.

        Returns
        -------
        A class with attributes self.data, self.properties and self.idefix.

        """

        self.data = file
        self.properties = file
        self.idefix = file

    @property
    def data(self):
        '''The attribute for accessing the data.'''

        return self.__data

    @data.setter
    def data(self, file):

        self.__data = add_data(file)

    @property
    def properties(self):
        '''The list of the names of the columns with shared building features.'''

        return self.__properties

    @properties.setter
    def properties(self, file):
        '''The columns that contain less unique values that there is buildings.'''

        df = self.data

        L = len(list(df.index.values))

        apu = []

        for column in df.columns:

            l = len(df[column].unique())

            if 1 < l < L:
                apu.append(column)

        self.__properties = apu

    @property
    def idefix(self):
        '''The list of the names of the columns with building spesific features.'''

        return self.__idefix

    @idefix.setter
    def idefix(self, file):
        '''The columns that contain an unique value per a building in the data set.'''

        df = self.data

        L = len(list(df.index.values))

        apu = []

        for column in df.columns:
            l = len(df[column].unique())
            if l == L:
                apu.append(column)

        self.__idefix = apu


class Pregraph(Buildingdata):

    def __init__(self, file: str):
        """
        A helper to calculate the adjacency matrix for the network of the buildings. 

        Parameters
        ----------
        file : string
            The path and filename of the building stock data to be analysized.

        Returns
        -------
        A class with the class Buildingdata as a parent class and added attribute self.distanceframe.

        """

        super().__init__(file)

        try:
            self.__distanceframe = np.loadtxt(
                f'distanceframe_{file}', delimiter=',').astype(float)

        except:
            self.distanceframe = self.data

    @property
    def distanceframe(self):
        '''The adjacency matrix, the distance between two buildings is the Jaccard distance between the sets of their properties.'''

        return self.__distanceframe

    @distanceframe.setter
    def distanceframe(self, data):

        L = len(list(data.index.values))

        data = data[self.properties].to_dict('index')

        distanceframe = np.zeros((L, L), dtype=np.float32)

        bt = combinations(range(L), 2)

        for pair in bt:
            p0 = set([(key, data[pair[0]][key]) for key in data[pair[0]]])
            p1 = set([(key, data[pair[1]][key]) for key in data[pair[1]]])
            distanceframe[pair[0]][pair[1]] = np.float64(dist(p0, p1))

        self.__distanceframe = distanceframe


class Network(Pregraph):

    def __init__(self, file: str):
        """
        The building network is a complete graph with the buildings as the nodes. Edge attributes are the shared properties of the buildings.

        Parameters
        ----------
        file : string
            The path and filename of the building stock data to be analysized.

        Returns
        -------
        A class with the class Pregraph as a parent class and added attributes self.network and self.clusters.
        """

        super().__init__(file)
        try:
            self.__network = pd.read_csv(f'nodelista_{file}')
        except:
            self.network = self.data
        try:
            self.__Z = np.loadtxt(f'Z_{file}', delimiter=',')
        except:
            self.Z = self.distanceframe

    @property
    def network(self):
        '''The edgelist of the network as a pandas.DataFrame.'''

        return self.__network

    @network.setter
    def network(self, data):
        '''The list of the buildingnodes'''

        self.__network = data[self.idefix]

    @property
    def Z(self):
        '''The hierarcical clustering of the buildingnodes using sklearn.AgglomerativeClustering(compute_full_tree=True, metric='precomputed', linkage='single', compute_distances=True) and precomputed distanceframe. Will try to set from memory.

        Parameters
        ----------

        self.distancematrix

        Returns
        -------
        self.Z the linkage matrix derived from self.distancematrix
        '''
        return self.__Z

    @Z.setter
    def Z(self, array):
        # Create linkage matrix
        model = AgglomerativeClustering(
            compute_full_tree=True, metric='precomputed', linkage='single', compute_distances=True)
        model = model.fit(self.distanceframe)
        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
                counts[i] = current_count

        linkage_matrix = np.column_stack(
            [model.children_, model.distances_, counts]).astype(float)

        '''
        G = pd.DataFrame(sch.fcluster(sch.single(array), 1.01*np.unique(array)[1], criterion='distance'), columns = [f'{np.unique(array)[1]}'])
        
        for i in np.unique(array)[2:]:
            G = pd.concat([G, pd.DataFrame(sch.fcluster(sch.single(array), i+0.01*np.unique(array)[1], criterion='distance'), columns = [f'{i}'])], axis=1)
        '''
        self.__Z = linkage_matrix


class Dendrogram(Network):

    def __init__(self, file: str, clustersize: int, clusternumber: int):
        super().__init__(file)
        self.__clustersize = clustersize
        self.__clusternumber = clusternumber
        try:
            self.__clusterings = pd.read_csv(f'clusterings_{file}')
        except:
            self.clusterings = file
        self.clusterdata = self.data

    @property
    def clusterings(self):
        '''

        Returns
        -------

        fcluster-class of scipy.cluster.hierarchy with criterion='maxclust' and n_clusters as .
        '''

        return self.__clusterings

    @clusterings.setter
    def clusterings(self, file):
        den, nodelist = to_tree(self.Z, rd=True)

        dist = 0
        for d in (n.dist for n in nodelist):
            clster = fcluster(self.Z, d, criterion='distance')
            labels, counts = np.unique(clster, return_counts=True)
            # (np.max(counts)+np.min(counts))/labels.shape[0]]
            cl = counts[counts > self.__clustersize]

            if cl.shape[0] > self.__clusternumber:
                dist = d
                break

        nl = [n for n in nodelist if n.dist >= d]
        n_clusters = len(nl[np.argmax(np.array([x[1]-x[0] for x in zip([n.dist for n in nodelist[:-2]
                         if n.dist >= dist], [n.dist for n in nodelist[1:] if n.dist >= dist])])) + 1:])

        self.__clusterings = fcluster(self.Z, n_clusters, criterion='maxclust')

    @property
    def clusterdata(self):
        '''

        Returns
        -------
        pandas.DataFrame

        '''

        return self.__clusterdata

    @clusterdata.setter
    def clusterdata(self, data):
        df = pd.concat([data[self.properties], self.clusterings], axis = 1)
        df.rename(columns={df.columns[-1]: 'clusterings'}, inplace=True)
        
        D = pd.DataFrame()

        labels, counts = np.unique(self.clusterings, return_counts=True)
        labels = list(map(int, list(labels))) 

        index = []
        # columns = labels
        index_names = ['property_name', 'property_values']
        column_names = ['clusters']

        Values = np.concatenate([pd.unique(df.loc[:, column].dropna())
                                for column in self.properties], axis=0)

        values = np.zeros((Values.shape[0], len(labels)))

        for column in self.properties:
            index += [(column, value)
                      for value in pd.unique(df.loc[:, column].dropna())]

            for label in labels:

                n = counts[labels.index(label)]

                S = df.loc[df['clusterings'] == label, :].value_counts(column)

                for value in [val for col, val in index if col == column]:

                    if value in S.index.values:
                        values[np.where(Values == value)[0][0]][labels.index(
                            label)] = round(S[value]/n, 1)
                    else:
                        values[np.where(Values == value)[0][0]
                               ][labels.index(label)] = None

        D = D.from_dict({'index': index, 'columns': labels, 'data': values,
                        'index_names': index_names, 'column_names': column_names}, orient='tight')

        self.__clusterdata = D


def main():
    '''
    Performed if __name__ == '__main__'.
    
    Parameters
    ----------

    sys.argv[1]: int, desired minimum clustersize.
    
    sys.argv[2]: int, desired maximum number of clusters with number of members over clustersize.
    
    sys.argv[3]: file name or file path of the building stock data.

    Returns
    -------
    Savefiles distanceframe_{filename}, nodelista_{filename}, Z_{filename}, clusterings_{filename}, Dendrogram.data_{filename}, dendrfig_{file[:-3]}pdf and tex_{file}[:-3]texb.


    '''

    file = sys.argv[3]
    clustersize = int(sys.argv[1])
    clusternumber = int(sys.argv[2])

    network = Dendrogram(file, clustersize, clusternumber)

    tallennus = f'distanceframe_{file}'

    np.savetxt(tallennus, network.distanceframe, delimiter=',')

    tallennus = f'nodelista_{file}'

    network.network.to_csv(tallennus)

    tallennus = f'Z_{file}'

    np.savetxt(tallennus, network.Z, delimiter=',')

    tallennus = f'clusterings_{file}'
    np.savetxt(tallennus, network.clusterings, delimiter=',')

    tallennus = f'Dendrogram.data_{file}'
    network.clusterdata.to_csv(tallennus)

    tallennus = f'dendrfig_{file[:-3]}pdf'
    dendrogram(network.Z, int(np.max(network.clusterings)), truncate_mode='lastp')
    plt.savefig(tallennus, format="pdf")

    tallennus = f'tex_{file[:-3]}tex'

    for value in (value for value in network.clusterdata.to_numpy().reshape(network.clusterdata.size) if value < clustersize):
        network.clusterdata.replace(value, np.nan, inplace=True)

    network.clusterdata.dropna(how='all').dropna(how='all', axis=1).to_latex(tallennus)


if __name__ == '__main__':
    # start = time.time()

    main()

    # stop = time.time()

    # duration = 2*40*(stop-start)/3600

    # print(f'Time {duration}')
