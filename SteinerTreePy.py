import pulp
from pulp import *
import numpy as np
import random
import scipy.sparse.csgraph as csg

# the following function reconstructs a path from the output predicessor of a scipy.sparse.csgraph.shortes_path 
# the function get_path is courtesy of https://stackoverflow.com/questions/53074947/examples-for-search-graph-using-scipy/53078901
def get_path(Pr, i, j):
    path = [j]
    k = j
    while Pr[i, k] != -9999:
        path.append(Pr[i, k])
        k = Pr[i, k]
    return path[::-1]


def uSTP(_V,_E,_T,_C, relaxation=False):
    """Given a graph G(_V, _E) with terminal set _T and the set of edge weights _C corresponding 1 to 1 with _E, return the adjacency matrix of the minimum stiener Tree Problem
    Parameters
    ----------
    _V : array
         An array of the labels corresponding to the vertices in G.
    _E : array
         An array of pairs of labels representing an edge between those two vertices
    _T : array
         An array of labels for the Terminal vertices of the graph
    _C : array
         An array of edge weights where _C[i] corresponds to the weight of _E[i]
    relaxation : Bool
         Indicates if ILP should be solved or relaxed LP with pulp
    
    """    

    # form constraints for the problem
    print("\n\n\n")
    n = len(_V)
    print("dimension of _V: ", n)
    m = len(_E)
    print("dimension of _E: ", m)
    card_t = len(_T) 
    
    
    # Generate mutually exclisive (ME) and collectivly exhaustive (CE) sets such that each set contains exactly one terminal node
    # Any sets formed in this way will satisfy the constraint that our terminals are connected if there is at least one edge connecting the set to its complement for our ILP.
    
    # to do this we construct an adjacency matrix for the graph
    
    # create array of vertices such that the index of that vertex into the array corresponds to the row/column that vertex represents in the adjacency matrix
    V = np.array(_V)
    assert(n == V.shape[0])
    print("V: ", V)
    
    T = np.array(_T)
    assert(card_t == T.shape[0])
    print("T: ", T)

    E = np.array(_E)
    assert(m == E.shape[0])
    print("E: \n", E)

    C = np.array(_C)
    assert(m == C.shape[0])
    print("C: ", C)
    
    #create zero matrix and fill with edges weights at indices of the vertices they are incident to
    adj_mat = np.zeros((n,n))
    #print("shape of adj_mat: ",adj_mat.shape)
    
    for e in range(m): 
        vert_1, vert_2 = _E[e]
        i = np.where(V == vert_1)[0][0]
        #print("i is: ", i)
        j = np.where(V == vert_2)[0][0]
        #print("j is: ", j)
        adj_mat[i][j] = _C[e]
        adj_mat[j][i] = _C[e]
    
    print("adj_mat: \n", adj_mat)
    
    # create sets to store subsets (ME & CE) of the graph as described above
    S = []
    visited = set()
    
    # enter loop to create ME & CE sets
    while(True):
        # if all terminals have their own sets break
        if (set(_T).issubset(visited)):
            break
        
        #get a random terminal from terminal set
        indx = random.randint(0, card_t - 1)
        t = T[indx]
        v = t
        
        #check if we've already made a set for that terminal
        if (t in visited):
            continue
        
        # this while is just a control flow like a goto, not actually functioning as a spinning loop
        while(True):
            #print("curr vertex: ", v)
            #check if v is visted if so break this while so it functions like a goto and not a loop -- this is messy but works
            if (v in visited):
                break
            
            # create set for this terminal, add to t to it and visted, this set will be added to S before we start a new set
            curr_set = set()
            visited.add(t)
            curr_set.add(t)
            #print("curr_set: ",curr_set)
            #print("visited: ",visited)
            # grab vertex adjacent to t and enter loop to expand set with adjacent non terminal nodes
            indx = np.where(V == t)[0][0]
            #print("index of terminal: ",indx)
            adj_list = adj_mat[indx]
            #print("adj_list: ", adj_list)
            v = V[np.nonzero(adj_list)[0][0]]
            #print("next vertex: ",v)
            while(True):

                # if visted break and select new random terminal
                if (v in visited):
                    S.append(curr_set)
                    break

                #check if grabed vertex is a terminal
                if (v in _T):
                    # we need to add set to S then break and "goto" the outer while
                    S.append(curr_set)
                    t = v
                    break
                
                # add to current set and visted
                curr_set.add(v)
                #print("curr_set: ", curr_set)
                visited.add(v)
                
                indx = np.where(V == v)[0][0]
                #print("index of node: ",indx)
                adj_list = adj_mat[indx]
                #print("adj_list: ", adj_list)
                v = V[np.nonzero(adj_list)[0][0]]
                #print("next vertex: ",v)
                
                # handshake lemma garentees that adj_list will contain at least one vertex that the vertex is connected to in adjacency matrix for an input connected graph 
                
    #print("S: ", S)

    # now one all sets are created there may still be vertices not part of a set, ie the sets are not collectivly exhaustive yet. As such we need to add all non visted vertices to a set they are connect to (which set does not matter) 
    for i in range(n):
        v = V[i]
        #print("current vertex: ", v)
        if (v in visited):
            continue
        
        # we need to find a set it is adjacent to. Since the graph is connected there must be one
        adj_list = adj_mat[i]
        #print("adj_list: ", adj_list)

        # add vertex to a partitioning set that it is adjacent to (doesn't matter which one)
        for curr_set in S:
            #find a nonzero (ie adjacent) entry in adj_list and see if it is in one of the sets. If so
            for i in np.nonzero(adj_list)[0]:
                if (V[i] in curr_set):
                    curr_set.add(v)
                    break
            if (v in curr_set): 
                break
                
    print("S: ", S)
    
    # we now have ME and CE sets which partition the graph s.t. each set contains exaclty one terminal
    
    # construct sets delta_curr_set which contain all edges where for any s1, s2 in a given set, the edge (s1,s2) is not in delta_set, and for any s3 in the complement of the given set, the edge (s1, s3) is in delta_set
    
    delta_S = []
    links_S = []
    
    # to do this we iterate through sets, and then add each edge that is connected to elements that are not in that set
    for curr_set in S:
        delta_curr_set = []
        links_curr_set = []
        for v in curr_set:
            # get the row corresponding to the vertex in the adj_matrix
            #print("current vertex: ", v)
            indx = np.where(V == v)[0][0]
            #print("index of current vertex: ", indx)
            adj_list = adj_mat[indx]
            #print("adj_list: \n", adj_list)
            #iterate through it's adjacencies to see edges connected to it
            for e_indx in np.nonzero(adj_list)[0]:
                v2 = V[e_indx]
                
                # if edge connects to an element outside of curr_set, then add the edge to delta_curr_set
                if (not(v2 in curr_set)):
                    if(not((v, v2) in delta_curr_set)):
                        delta_curr_set.append((v, v2))
                    if(not(v in links_curr_set)):
                        links_curr_set.append(v)
    
        delta_S.append(delta_curr_set)
        links_S.append(links_curr_set)
        
    print("delta_S: ", delta_S)
    print("links_S: ", links_S)

    # generate shortest paths from vertices to edges in a delta_set to the terminal in their respective set
    paths_S = []
    path_weights_S = []
    for i in range(len(S)): 
        # get adjacency matrix for the subgraph corresponding to the partitioning set S. This can be done by selecting the subset of the adjaceny matrix that exists inside the partition
        
        #retreive the current set
        curr_set = S[i]
        #print("curr_set: ", curr_set)
        
        #retreive the links for that set to calculate shortest paths from
        curr_links = links_S[i]
        #print("links for curr_set: ", links_S[i])

        #get the indices of the nodes in curr_set to form sub adjacency matrix for the partition
        indices = []
        index_of_terminal = -m-1 # negative m+1 is put here as a place holder since it doesn't exist in the matrix, since each set must contain a terminal this will be overwritten
        for node in curr_set:
            indx = np.where(V == node)[0][0]
            if (node in T):
                index_of_terminal = indx
            #print("index of current node: ", indx)
            indices.append(indx)
        #print("indices: ", indices)

        # build adjacency matrix for connections within the set
        adj_sub_mat = adj_mat[indices, :][:, indices]
        #print("sub graph adjacency matrix: \n", adj_sub_mat)
        
        #get the indices of the links in the new adjacency matrix
        curr_links_indices = []
        for i in range(len(indices)):
            for l in curr_links: 
                if (indices[i] == np.where(V == l)[0][0]):
                    curr_links_indices.append(i)
        #print("curr_links_indices: ", curr_links_indices)

        # compute shortest paths using Dijkstra's algorithm from curr_links and predecessor matrix used to reconstruct the paths
        short_path, predecessors = csg.shortest_path(adj_sub_mat, method='D', directed=False, indices= curr_links_indices, return_predecessors=True)
        #print("shortest_path on ", curr_set, ": \n", short_path)
        #print("predicessors: \n", predecessors)


        # get converted index of the terminal
        converted_terminal_indx = np.where(indices == index_of_terminal)[0][0]
        #print("converted_terminal_index: ", converted_terminal_indx)
        
        # recover shortest paths from the predecessors output of scipy 'csg' shortest_path funciton and convert to origional labels
        curr_paths = []
        for i in range(len(curr_links_indices)):
            # get path in terms of index into indices
            rel_path = get_path(predecessors, i, converted_terminal_indx)
            #print("path relative to indices of sub adjacency matrix: ", rel_path)
            path = []
            for i in range(len(rel_path)):
                if (i < len(rel_path) - 1):
                    path.append((V[indices[rel_path[i]]], V[indices[rel_path[i+1]]]))
                else: 
                    if (i == 0):
                        # this is then a terminal
                        terminal = V[indices[converted_terminal_indx]]
                        path.append((terminal, ))
            curr_paths.append(path)
        paths_S.append(curr_paths)

        # get the shortest paths from terminal to each link in the set
        shortest_paths_to_terminal = short_path[:, converted_terminal_indx].T
        #print("shortest_paths_to_terminal: ", shortest_paths_to_terminal)
        path_weights_S.append(shortest_paths_to_terminal)
    print("paths_S: ", paths_S)
    print("paths_weights: ", path_weights_S)

    # we now have all elements needed for our formulation of the ILP

    # formulate the ILP and solve with pulp
    
    #Decision Binary Variables (Non-negative variables between 0 and 1 if relaxation=True)
    if (not(relaxation)):
        x = {i : LpVariable(name=f"x{i}", cat="Binary") for i in range(m)}
    else:
        x = {i : LpVariable(name=f"x{i}", lowBound=0, upBound=1) for i in range(m)}

    # Problem definitition and initialization
    p_stp = LpProblem("Minimum_Steiner_Tree_Problem", LpMinimize)

    #Objective Function
    p_stp += lpSum([x[i]*C[i] for i in range(m)])


    #Constraints

    #create set to hold all unique mapped_delta_S, since we need greater or equal to num_sets-1 connections between partitions for them to be connected
    mapped_delta_S = set()
    for i in range(len(S)):
        #curr_set = S[i]
        delta_curr_set = delta_S[i]
        links_curr_set = links_S[i]
        curr_paths = paths_S[i]
        curr_path_weights = path_weights_S[i]

        #map edges in delta_curr_set and curr_paths to x variables by getting index into E of that edge, noting that order matters for index so try other way if not in E
        #and add constraints to Problem       
        delta_curr_set_mapped = []
        delta_links_curr_set = []
        for edge in delta_curr_set:
            # due to construction of edges in delta, the first value of the pair will be the link vertex

            try:
                mapped_edge = _E.index(edge)
            except ValueError:
                mapped_edge = _E.index((edge[1], edge[0]))

            #print("edge index: ", mapped_edge)
            delta_curr_set_mapped.append(mapped_edge)
            linked_vertex_indx = links_curr_set.index(edge[0])
            #print("incident_vertex indx: ", linked_vertex_indx)
            delta_links_curr_set.append(linked_vertex_indx)
            mapped_delta_S.add(mapped_edge)

        # constraint that at least one edge into the set must exist for terminal to be connected
        #print("delta_curr_set_mapped: ", delta_curr_set_mapped)
        #print("delta_links_curr_set: ", delta_links_curr_set)
        #for bleh in delta_curr_set_mapped:
        #    print(x[bleh])
        p_stp += lpSum(x[k] for k in delta_curr_set_mapped) >= 1
        #print(p_stp)

        
        for path in curr_paths:
            #print("path: ", path)
            mapped_path = []
            for edge in path:
                try:
                    mapped_edge = _E.index(edge)
                    mapped_path.append(mapped_edge)
                    #print("edge index: ", mapped_edge)
                except ValueError:
                    try:
                        mapped_edge = _E.index((edge[1], edge[0]))
                        mapped_path.append(mapped_edge)
                        #print("edge index: ", mapped_edge)
                    except IndexError:
                        # this means no path was needed since the linked vertex was a terminal
                        print(edge)
                    
            #print("mapped_path: ", mapped_path)

            #each path is correlated with a linking vertex since paths are generated from them to a terminal, so if the index of a link is the same index as the path into paths, then any delta incident to that link would trigger the path
            for l in range(len(delta_links_curr_set)):
                if (curr_paths.index(path) == delta_links_curr_set[l]) :

                    #constraint that if an edge in delta_curr_set is used then the entire path connecting that linked vertex to terminal must also be used so that terminal is connected
                    p_stp += lpSum([x[p]*C[p]] for p in mapped_path) >= x[delta_curr_set_mapped[l]] * curr_path_weights[curr_paths.index(path)]
    
    print("mapped_delta_S: ", mapped_delta_S)
    #constraint that at least num_sets-1 connections connect the partitions so that we don't get disjoint output
    p_stp += lpSum(x[i] for i in mapped_delta_S) >= (len(S)-1)/2 + 1
    #p_stp += lpSum(x[i] for i in mapped_delta_S) >= len(S)-1


    #solve the stp with pulp
    print(p_stp)

    p_stp.solve()
    print("status: ", LpStatus[p_stp.status])

    print("Minimal Weights = ", value(p_stp.objective))

    for var in p_stp.variables():
        print(var.name, " = ", var.varValue)
        


        
        
        

        







# test for standard graph
V_set = [1,2,3,4,5,6,7,8]
E_set = [(1,2),(1,4),(2,7),(2,4),(2,3),(3,6),(3,5),(4,8)]
T_set = [2,4,5,6]
C_set = [7,3,2,11,8,2,1,4]

uSTP(V_set, E_set, T_set, C_set)

# test with string labels
V_set = ['a','b','c','d','e','f','g','h', 'i']
E_set = [('a','b'),('b','c'),('a','e'),('e','f'),('c','f'),('c','d'),('f','g'),('f','h'),('d', 'h'), ('h', 'c'), ('i', 'h')]
T_set = ['a','d','h']
C_set = [1, 7, 3, 9, 4, 2, 3, 2, 1, 1, 1]

uSTP(V_set, E_set, T_set, C_set)

# test for cyclic graph