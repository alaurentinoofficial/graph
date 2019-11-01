
# coding: utf-8

# In[62]:


import numpy as np


# <br>
# <br>
# 
# ## Implements the abstraction

# In[63]:


class GenericGraph:
    def __init__(self, initial_graph=None, dtype='LIST'):
        self._graph_root = None
        
        if initial_graph != None:
            if dtype == 'LIST':
                self._graph_root = GenericGraph.to_list(initial_graph)
                self._dtype = 'LIST'
            else:
                self._graph_root = GenericGraph.to_matrix(initial_graph)
                self._dtype = 'MATRIX'

    def add_vertice(self):
        raise ValueError("This class is an Abstract class and don't has implemented this method!\nPlease extend this class and implements!")
    
    def add_edge(self, a, b, value=1, oriented=False):
        raise ValueError("This class is an Abstract class and don't has implemented this method!\nPlease extend this class and implements!")
    
    def adjacent(self, index, with_values=False):
        raise ValueError("This class is an Abstract class and don't has implemented this method!\nPlease extend this class and implements!")
    
    def output_degree(self, index):
        raise ValueError("This class is an Abstract class and don't has implemented this method!\nPlease extend this class and implements!")
    
    def input_degree(self, index):
        raise ValueError("This class is an Abstract class and don't has implemented this method!\nPlease extend this class and implements!")
    
    def lower_edge(self, index):
        raise ValueError("This class is an Abstract class and don't has implemented this method!\nPlease extend this class and implements!")
    
    def higher_edge(self, index):
        raise ValueError("This class is an Abstract class and don't has implemented this method!\nPlease extend this class and implements!")
        
    @staticmethod
    def to_matrix(tuples):
        if tuples != None:
            tuples = np.array(tuples)
            max_value = tuples[:, 0:2].max() + 1
            graph = np.zeros((max_value, max_value,))

            if tuples.shape[1] == 2:
                for a, b in tuples:
                    graph[a][b] = 1

            elif tuples.shape[1] == 3:
                for a, b, c in tuples:
                    graph[a][b] = c

            else:
                graph = None
        
            return graph
        
        return None
    
    @staticmethod
    def to_list(tuples):
        if tuples != None:
            tuples = np.array(tuples)
            max_value = tuples[:, 0:2].max() + 1
            graph = [[] for _ in range(max_value)]

            if tuples.shape[1] == 2:
                for a, b in tuples:
                    graph[a].append([b, 1])

            elif tuples.shape[1] == 3:
                for a, b, c in tuples:
                    graph[a].append([b, c])  
            return graph
        return None
    
    def search_depth(self, index=0, already_visited=[]):
        if index not in already_visited:
            already_visited.append(index)
            next_vertices = self.adjacent(index)

            for item in next_vertices:
                result = self.search_depth(item, already_visited[:])

                if len(result) > len(already_visited):
                    already_visited = result
        
        return already_visited
    
    def search_width(self):
        index=0
        already_visited=set()
        maxx=[]
        
        already_visited.add(index)
        sons = self.adjacent(index)
        
        if len(sons) > len(maxx):
            maxx = sons
            
        while True:
            sub_sons = []
            for i in sons:
                if i not in already_visited:
                    already_visited.add(i)
                    sub_sons += [item for item in self.adjacent(i) if item not in already_visited]
            
            for i in sub_sons:
                already_visited.add(i)

            if len(sub_sons) > len(maxx):
                maxx = sub_sons

            sons = sub_sons

            if len(sons) == 0:
                break
            
        
        return maxx


# <br>
# <br>
# 
# ## Create a List graph

# In[64]:


class ListGraph(GenericGraph):
    def __init__(self, initial_graph):
        super().__init__(initial_graph, dtype='LIST')
    
    def __len__(self):
        return len(self._graph_root)
    
    def __str__(self):
        return "\n".join(["{}: {}".format(i, str(self._graph_root[i])) for i in range(len(self._graph_root))])
    
    def __repr__(self):
        return "Graph.ListGraph()"
    
    def __getitem__(self, index):
        return tuple(self.adjacent(index))
    
    def add_vertice(self):
        self._graph_root.append([])
        
    @staticmethod
    def index_in(sub, index):
        for vertice in sub:
            if vertice[0] == index:
                return True
        
    @staticmethod
    def add_if_not_exists_or_replace(graph, a, b, value):
        sub = graph[a]
        
        exists = -1
        for i in range(len(sub)):
            if sub[i][0] == b:
                exists = i
                break
        
        if exists == -1:
            graph[a].append([b, value])
        else:
            graph[a][exists][1] = value
    
    def add_edge(self, a, b, value=1, oriented=False):
        ListGraph.add_if_not_exists_or_replace(self._graph_root, a, b, value)
        
        if not oriented:
            ListGraph.add_if_not_exists_or_replace(self._graph_root, b, a, value)
    
    def adjacent(self, index, with_values=False):
        if with_values:
            return self._graph_root[index]
        
        return [item[0] for item in self._graph_root[index] if len(item) > 0]
    
    def output_degree(self, index):
        return len(self._graph_root[index])
    
    def input_degree(self, index):
        result = 0
        
        for i in range(self.__len__()):
            if i != index and ListGraph.index_in(self._graph_root[i], index):
                result += 1
                
        return result
    
    def lower_edge(self, index):
        result = []
        adj = sorted(self._graph_root[index], key=lambda x: x[1])
        
        if len(adj) > 0:
            minx = adj[0][1]
            result = list(filter(lambda x: x[1] == minx, adj))
        
        return result
    
    def higher_edge(self, index):
        result = []
        adj = sorted(self._graph_root[index], key=lambda x: x[1], reverse=True)
        
        if len(adj) > 0:
            maxx = adj[0][1]
            result = list(filter(lambda x: x[1] == maxx, adj))
        
        return result


# In[65]:


graph = ListGraph([(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0), (1, 7), (7, 1), (2, 4), (4, 2), (2, 5), (5, 2), (3, 6), (6, 3), (4, 8)])

print("Before:")
print(graph)
print()

graph.add_edge(4, 8, 1)

print("After:")
print(graph)
print()

for i in range(len(graph)):
    print("[{}] Adjacent: {}".format(i, graph[i] ))
    print("[{}] Input = {}\tOutput = {}".format(i, graph.input_degree(i), graph.output_degree(i)))
    print("[{}] Lower vertice = {}".format(i, graph.lower_edge(i)))
    print("[{}] Higher vertice = {}".format(i, graph.higher_edge(i)))
    print()

print()
print("Depth: ", graph.search_depth())
print("Width: ", graph.search_width())
print("\n" + "=" * 50 + "\n")


# <br>
# <br>
# 
# ## Create a matrix graph

# In[66]:


class MatrixGraph(GenericGraph):
    def __init__(self, initial_graph):
        super().__init__(initial_graph, dtype='MATRIX')
        
    def __len__(self):
        return self._graph_root.shape[0]
        
    def __str__(self):
        letter_spacing = len(str(self._graph_root.max()))
        
        add_space = lambda spacing, x: " " * (spacing - len(x)) + x

        result = add_space(letter_spacing, "") + "".join([add_space(letter_spacing, str(item)) for item in range(self._graph_root.shape[1])]) + "\n"
        result += "\n".join([str(index) + " " * letter_spacing + str(line) for index, line in enumerate(list(self._graph_root))])
        
        return result
    
    def __repr__(self):
        return "Graph.MatrixGraph()"
        
    def __getitem__(self, index):
        return tuple(self.adjacent(index))
    
    def add_vertice(self):
        self._graph_root = np.pad(self._graph_root, [(0, 1), (0, 1)], mode='constant')
    
    def add_edge(self, a, b, value=1, oriented=False):
        self._graph_root[a, b] = value
        
        if not oriented:
            self._graph_root[b, a] = value
    
    def input_degree(self, index):
        return (self._graph_root[:, index] != 0).sum()
    
    def output_degree(self, index):
        return (self._graph_root[index, :] != 0).sum()
    
    def adjacent(self, index, with_values=False):
        if with_values:
            return list(set(filter(lambda x: x[1] > 0, enumerate(self._graph_root[index, :]))))
        
        return list(set(np.where(self._graph_root[index, :] > 0)[0]))
    
    def lower_edge(self, index):
        result = []
        adj = self.adjacent(index, with_values=True)
        adj = sorted(adj, key=lambda x: x[1])
        
        if len(adj) > 0:
            minx = adj[0][1]
            result = list(filter(lambda x: x[1] == minx, adj))
        
        return result

    def higher_edge(self, index):
        result = []
        adj = self.adjacent(index, with_values=True)
        adj = sorted(adj, key=lambda x: x[1], reverse=True)
        
        if len(adj) > 0:
            maxx = adj[0][1]
            result = list(filter(lambda x: x[1] == maxx, adj))
        
        return result
    
    def search_depth(self, index=0, already_visited=[]):
        if index not in already_visited:
            already_visited.append(index)
            next_vertices = self.adjacent(index)

            for item in next_vertices:
                result = self.search_depth(item, already_visited[:])

                if len(result) > len(already_visited):
                    already_visited = result
        
        return already_visited


# In[67]:


graph = MatrixGraph([(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0), (1, 7), (7, 1), (2, 4), (4, 2), (2, 5), (5, 2), (3, 6), (6, 3), (4, 8)])

print("Before:")
print(graph)
print()

graph.add_edge(4, 8, 1)

print("After:")
print(graph)
print()

for i in range(len(graph)):
    print("[{}] Adjacent: {}".format(i, graph[i] ))
    print("[{}] Input = {}\tOutput = {}".format(i, graph.input_degree(i), graph.output_degree(i)))
    print("[{}] Lower vertice = {}".format(i, graph.lower_edge(i)))
    print("[{}] Higher vertice = {}".format(i, graph.higher_edge(i)))
    print()

print()
print("Depth: ", graph.search_depth())
print("Width: ", graph.search_width())

