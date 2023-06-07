import networkx as nx
import numpy as np
import torch

class Embedding():
    def embed_action(self,nodes):
        raise NotImplementedError

class HeuristicActionEmbedding(Embedding):
    def __init__(self, graph,thresholds):
        self.graph = graph
        self.num_nodes = self.graph.number_of_nodes()
        self.num_features = self.num_nodes
        # self.coupled_features = True
        # self.num_features = 0
        # self.num_nodes = self.graph.number_of_nodes()
        # #Get Normalized Nodes
        # self.min_node = 0
        # self.max_node = self.num_nodes-1
        # self.normalized_nodes = self.normalize(torch.tensor([n for n in range(self.max_node+1)]),self.min_node,self.max_node)
        # self.num_features += self.num_nodes

        # # Calculate degree centrality
        # degree_centrality = nx.degree_centrality(self.graph)
        # self.min_degree_centrality = min(degree_centrality.values())
        # self.max_degree_centrality = max(degree_centrality.values())
        # degree_centrality = torch.tensor([degree_centrality[i] for i in range(self.max_node+1)])
        # self.normalized_degree_centrality = self.normalize(degree_centrality,self.min_degree_centrality,self.max_degree_centrality)

        # # Calculate betweenness centrality
        # betweenness_centrality = nx.betweenness_centrality(self.graph)
        # min_c = min(betweenness_centrality.values())
        # max_c = max(betweenness_centrality.values())
        # betweenness_centrality = torch.tensor([betweenness_centrality[i] for i in range(self.max_node+1)])
        # self.normalized_betweenness_centrality = self.normalize(betweenness_centrality,min_c,max_c)
        # self.num_features += 2*self.num_nodes
        # #Get Normalized Thresholds
        # self.thresholds = thresholds
        # if len(self.thresholds) > 0:
        #     min_t= min(self.thresholds)
        #     max_t = max(self.thresholds)
        #     self.normalized_thresholds = self.normalize(torch.tensor(self.thresholds),min_t,max_t)
            # self.neighbor_mean_thresholds = torch.zeros(self.graph.number_of_nodes())
            # self.neighbor_min_thresholds = torch.zeros(self.graph.number_of_nodes())
            # self.neighbor_mean_degree = torch.zeros(self.graph.number_of_nodes())
            # self.neighbor_min_degree = torch.zeros(self.graph.number_of_nodes())
            #self.neighbors = []
            # for n in range(self.graph.number_of_nodes()):
            #     neighbor_thresholds = []
            #     neighbor_degrees = []
            #     two_hop_neighbors = set()
            #     self.neighbors.append(list(self.graph.neighbors(n)))
            #     for i,v in enumerate(self.neighbors[n]):
            #         neighbor_thresholds.append(self.normalized_thresholds[v])
            #         neighbor_degrees.append(self.normalized_degree_centrality[v])

            #     neighbor_thresholds = torch.tensor(neighbor_thresholds)
            #     self.neighbor_mean_thresholds[n] = torch.mean(neighbor_thresholds)
            #     self.neighbor_min_thresholds[n] = torch.min(neighbor_thresholds)
            #     neighbor_degrees = torch.tensor(neighbor_degrees)
            #     self.neighbor_mean_degree[n] = torch.mean(neighbor_degrees)
            #     self.neighbor_min_degree[n] = torch.min(neighbor_degrees)

            #self.num_features += self.num_nodes



        # # Calculate closeness centrality
        # closeness_centrality = nx.closeness_centrality(self.graph)
        # min_c = min(closeness_centrality.values())
        # max_c = max(closeness_centrality.values())
        # closeness_centrality = torch.tensor([closeness_centrality[i] for i in range(self.max_node+1)])
        # self.normalized_closeness_centrality = self.normalize(closeness_centrality,min_c,max_c)


        # # Calculate eigenvector centrality
        # eigenvector_centrality = nx.eigenvector_centrality(self.graph)
        # min_c = min(eigenvector_centrality.values())
        # max_c = max(eigenvector_centrality.values())
        # eigenvector_centrality = torch.tensor([eigenvector_centrality[i] for i in range(self.max_node+1)])
        # self.normalized_eigenvector_centrality = self.normalize(eigenvector_centrality,min_c,max_c)

    def embed_action(self, nodes):
        B = nodes.shape[0]
        features = torch.zeros(B,self.num_features)
        #multi-hot encoding of nodes targeted
        binary_mask = torch.zeros((nodes.size(0), self.num_nodes))
        binary_mask.scatter_(1,nodes,1)
        features = self.normalize(binary_mask,0,1)
        #features[:,0:self.num_nodes] = 
        # # Calculate centrality measures for the given nodes

        #features[:,2:4] = self.normalized_degree_centrality[nodes.flatten()].reshape(nodes.shape)
        # features[:,4:6] = self.neighbor_mean_degree[nodes.flatten()].reshape(nodes.shape)
        # features[:,6:8] = self.neighbor_min_degree[nodes.flatten()].reshape(nodes.shape)

        # features[:,4:6] = self.normalized_closeness_centrality[nodes.flatten()].reshape(nodes.shape)
        # features[:,6:8] = self.normalized_betweenness_centrality[nodes.flatten()].reshape(nodes.shape)
        # features[:,8:10] = self.normalized_eigenvector_centrality[nodes.flatten()].reshape(nodes.shape)
        # i = 8
        # if len(self.thresholds) > 0:
        #     features[:,i:i+2] = self.neighbor_mean_thresholds[nodes.flatten()].reshape(nodes.shape)
        #     features[:,i+2:i+4] = self.neighbor_mean_thresholds[nodes.flatten()].reshape(nodes.shape)
        # i += 4

        # # Calculate distance between nodes
        # if self.coupled_features:
        #     distance = [nx.shortest_path_length(self.graph, source=list(self.graph.nodes)[nodes[j,0].item()], 
        #         target=list(self.graph.nodes)[nodes[j,1].item()]) for j in range(B)]
        #     normalized_distance = self.normalize(torch.tensor(distance), 1, self.max_degree_centrality*self.max_node)
        #     features[:,i] = normalized_distance
        #     i +=1

        #     common_neighbors = [list(nx.common_neighbors(self.graph,list(self.graph.nodes)[nodes[j,0].item()],
        #         list(self.graph.nodes)[nodes[j,1].item()])) for j in range(B)]

        #     num_common_neighbors = torch.zeros(B)
        #     mean_thresh_common_neighbors = torch.zeros(B)
        #     min_thresh_common_neighbors = torch.zeros(B)

        #     for k,cn in enumerate(common_neighbors):
        #         num_common_neighbors[k] = len(cn)
        #         degree_cn = torch.tensor([self.normalized_thresholds[j] for j in cn])
        #         mean_thresh_common_neighbors[k] = torch.mean(degree_cn)
        #         min_thresh_common_neighbors[k] = torch.min(degree_cn) if degree_cn.numel() > 0 else 0

        #     normalized_num_common_neighbors = self.normalize(num_common_neighbors,0,np.log(self.max_node+1))
        #     features[:,i] = normalized_num_common_neighbors
        #     features[:,i+1] = mean_thresh_common_neighbors
        #     features[:,i+1] = min_thresh_common_neighbors
        return features

    def normalize(self, values, minimum, maximum):
        if maximum - minimum == 0:
            return torch.zeros_like(values)
        return -1.0 + 2.0 * (values - minimum*torch.ones_like(values)) / (maximum - minimum)

class GNN_Embedding(Embedding):
    def __init__(self, graph, model):
        self.graph = graph
