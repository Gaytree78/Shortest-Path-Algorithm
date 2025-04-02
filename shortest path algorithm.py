import networkx as nx
import matplotlib.pyplot as plt
import heapq


graph = {
    'A': [('B', 4), ('C', 1)],
    'B': [('D', 1)],
    'C': [('B', 2), ('D', 5)],
    'D': []
}


G = nx.DiGraph()
for node in graph:
    for neighbor, weight in graph[node]:
        G.add_edge(node, neighbor, weight=weight)


def draw_graph(G, shortest_paths=None, title="Graph Visualization"):
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')
    plt.figure(figsize=(6, 4))

    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            edge_color='gray', node_size=1500, font_size=12)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    if shortest_paths:
        nx.draw_networkx_edges(
            G, pos, edgelist=shortest_paths, edge_color="red", width=2)

    plt.title(title)
    plt.show()


draw_graph(G, title="Original Graph with Edge Weights")


def dijkstra(graph, start):
    shortest_paths = {node: float('inf') for node in graph}
    shortest_paths[start] = 0
    pq = [(0, start)]
    parent = {}

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        for neighbor, weight in graph.get(current_node, []):
            distance = current_distance + weight
            if distance < shortest_paths[neighbor]:
                shortest_paths[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
                parent[neighbor] = current_node

    path_edges = [(parent[node], node) for node in parent]

    return shortest_paths, path_edges


_, dijkstra_edges = dijkstra(graph, 'A')
draw_graph(G, dijkstra_edges, title="Dijkstra Shortest Path Visualization")


def bellman_ford(graph, start):
    shortest_paths = {node: float('inf') for node in graph}
    shortest_paths[start] = 0
    parent = {}

    for _ in range(len(graph) - 1):
        for node in graph:
            for neighbor, weight in graph[node]:
                if shortest_paths[node] + weight < shortest_paths[neighbor]:
                    shortest_paths[neighbor] = shortest_paths[node] + weight
                    parent[neighbor] = node

    path_edges = [(parent[node], node) for node in parent]

    return shortest_paths, path_edges


_, bellman_edges = bellman_ford(graph, 'A')
draw_graph(G, bellman_edges, title="Bellman-Ford Shortest Path Visualization")
