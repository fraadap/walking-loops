import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_or_create_graph(place="Wemmel, Belgium", num_pois=50, radius_meters=3000, cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    
    safe_place_name = "".join([c if c.isalnum() else "_" for c in place.lower()])
    graph_path = os.path.join(cache_dir, f"graph_{safe_place_name}_{radius_meters}m.graphml")
    # Cambiato suffisso in 'precomputed' per forzare una nuova cache pulita
    info_path = os.path.join(cache_dir, f"graph_info_{safe_place_name}_{radius_meters}m_precomputed.pkl")
    
    # 1. LOAD DA CACHE
    if os.path.exists(graph_path) and os.path.exists(info_path):
        print(f"Loading graph and precomputed paths from cache for {place}...")
        G = ox.load_graphml(graph_path)
        with open(info_path, "rb") as f:
            home_node, pois = pickle.load(f) # Non carichiamo più i path_lengths enormi
        return G, home_node, pois
    
    # 2. DOWNLOAD MAPPA
    print(f"Downloading graph for {place} with {radius_meters}m radius...")
    center_coords = ox.geocode(place)
    G = ox.graph_from_point(center_coords, dist=radius_meters, network_type="walk")
    
    G = list(G.subgraph(c) for c in nx.strongly_connected_components(G))
    G = max(G, key=len)
    nodes = list(G.nodes)
    np.random.seed(42)  
    
    home_node = np.random.choice(nodes)
    
    # 3. SELEZIONE POI
    print(f"[{place}] Calculating spatial distribution for POIs...")
    distances = nx.single_source_dijkstra_path_length(G, home_node, weight='length')
            
    sorted_nodes = sorted(distances.keys(), key=lambda n: distances[n])
    chunk_size = max(1, len(sorted_nodes) // num_pois)
    pois = []
    
    for i in range(num_pois):
        chunk = sorted_nodes[i*chunk_size : (i+1)*chunk_size]
        if home_node in chunk:
            chunk.remove(home_node)
        if len(chunk) > 0:
            pois.append(np.random.choice(chunk))
        else:
            pois.append(np.random.choice(nodes))
            
    # SHUFFLE DISTRUTTIVO
    np.random.shuffle(pois)
    pois = pois.tolist() if isinstance(pois, np.ndarray) else list(pois)
            
    # 4. PRECALCOLO PATHS (Ottimizzazione Single-Source su soli 51 nodi)
    print(f"[{place}] Precomputing all paths between key nodes (Fast Mode)...")
    key_nodes = [home_node] + pois
    paths = {}
    path_lengths = {}
    
    for u in tqdm(key_nodes, desc=f"Routing {place}"):
        paths[u] = {}
        path_lengths[u] = {}
        
        # Lancia la macchia d'olio da 'u'
        lengths, all_paths = nx.single_source_dijkstra(G, u, weight='length')
        
        # Filtra solo i nodi che ci interessano (evitando di salvare decine di migliaia di strade inutili)
        for v in key_nodes:
            if v in all_paths:
                paths[u][v] = all_paths[v]
                path_lengths[u][v] = lengths[v]
            else:
                paths[u][v] = []
                path_lengths[u][v] = float('inf')

    # 5. SALVATAGGIO IN CACHE
    ox.save_graphml(G, graph_path)
    with open(info_path, "wb") as f:
        pickle.dump((home_node, pois), f) # Salviamo solo Casa e POI
        
    return G, home_node, pois, paths, path_lengths

def plot_graph(G):
    """Plot the graph using osmnx and save to file."""
    fig, ax = ox.plot_graph(G, show=False, close=False)
    plt.savefig("graph_plot.png")
    print("Graph plot saved to graph_plot.png")

if __name__ == "__main__":
    G, home_node, pois, paths, path_lengths = get_or_create_graph(place="Brussels, Belgium", num_pois=80, radius_meters=3000)
    plot_graph(G)