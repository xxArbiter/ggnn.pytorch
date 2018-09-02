import numpy as np
import scipy.io as spio

def loadGraph():
    graph = []
    with open('test_data/graph_1.csv', 'r') as f:
        for line in f:
            outEdge = line[:-1].split(' ')
            outEdge = outEdge[1:]
            for i in range(len(outEdge)):
                outEdge[i] = int(outEdge[i])
            graph.append(outEdge)
    return graph

def formAdjMatrix(graph):
    dim = len(graph)
    A = np.zeros([dim, dim])
    for i in range(dim):
        for j in graph[i]:
            A[i, j] = 1
    At = A.transpose()
    A = np.concatenate((A, At), axis=1)
    return A

def loadData():
    data = spio.loadmat('test_data/data_1.mat')
    data = data['data']
    data /= 100
    return data

def dataConvert(data):
    dataList = []
    for t in range(data.shape[1] - 144):    # T - truncate
        dataList.append([data[:, t:t+144], data[:, t+144]]) # [annotation[144,n], target[1,n]]
    return dataList

def splitSet(data):
    L = len(data)
    trainData = []
    valData = []
    for i in range(L):
        if i <= int(L*0.8):
            trainData.append(data[i])
        else:
            valData.append(data[i])
    return trainData, valData

class bAbIDataset():
    """
    Load bAbI tasks for GGNN
    """
    def __init__(self, is_train):
        graph = loadGraph()
        self.A = formAdjMatrix(graph)
        allData = loadData()[0, :, :, 0]   # [T, n]
        allData = allData.transpose()      # [n, T]
        self.n_edge_types =  1
        self.n_tasks = 1
        self.n_node = allData.shape[0]

        allData = dataConvert(allData)
        trainData, valData = splitSet(allData)

        if is_train:
            self.data = trainData
        else:
            self.data = valData

    def __getitem__(self, index):
        am = self.A
        annotation = self.data[index][0]
        target = self.data[index][1]
        return am, annotation, target

    def __len__(self):
        return len(self.data)

