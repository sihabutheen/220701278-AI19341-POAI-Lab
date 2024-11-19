import networkx as nx

#FUNCTION TO SOLVE DFS
def solveDFS(graph, v, visited) :
  visited.add(v) 
  print(v, end=' ') 
  for neighbour in graph[v] : 
    if neighbour not in visited : 
      solveDFS(graph, neighbour, visited) 
      
g = nx.DiGraph() 

#CREATE A GRAPH USING NETWORKX
g.add_edges_from([('A','B'),('A','C'),('C','G'),('B','D'),('B','E'),('D','F'),('A','E')])
# Add edges for that graph
nx.draw(g, with_labels=True) # Graph Visualization

#SOLVE DFS FOR THAT GRAPH
print("Following is DFS from (starting from vertex A)")
visited = set()
solveDFS(g, 'A', visited)
