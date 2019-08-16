import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import permutations


def get_graph():
    print('Enter vertex names, split them with spaces; type "Random vertex_number edge_density_level(1-4) + args"'
          ' for a random graph(mpre info in comments below this line)')
    #Random vertex_number edge_density_level(1-4) is_weighted is_directed; set "random" for a random result
    vertex_names = input().split()
    if len(vertex_names) == 0:
        return 'Empty input'
    if vertex_names[0] == 'Random':
        vertex_names, vertex_number, is_weighted, is_directed, adj_list = make_random_graph(vertex_names)
    else:
        for i in vertex_names:
            if vertex_names.count(i) > 1:
                print('You are very bad person')
                return 'GO TO HELL'
        vertex_number = len(vertex_names)

        print('Is graph directed?(y/n)')
        is_directed = False
        directed = input()
        if directed == 'y' or directed == 'yes':
            is_directed = True

        print('Enter edges like this: vertex1,vertex2,distance_between; if not weighted: vertex1,vertex2')
        adj_list = input().split()

        is_weighted = True
        weight_control = adj_list[0].split(',')
        if len(weight_control) == 2:
            is_weighted = False
    return vertex_names,adj_list,is_weighted,is_directed

def make_random_graph(args):
    vertex_names = ''
    adj_list = []
    vertex_number = random.randint(1, 16)
    if len(args) > 1 and args[1] != 'random':
            vertex_number = int(args[1])
    vertex_names = ['vertex'+str(j+1) for j in range(vertex_number)]

    is_directed = random.choice([True,False])
    is_weighted = random.choice([True,False])

    if len(args) > 3 and args[3] != 'random':
        is_weighted = False if args[3] == 'False' else True
    if len(args) > 4 and args[4] != 'random':
        is_directed = False if args[4] == 'False' else True

    if len(args) > 2:
        if args[2] == 'random':
            edge_density = random.randint(1,4)
        else:
            edge_density = args[2]

    if not is_directed:
        edges_pool = []
        if len(args) < 2:
            edges_number = random.randint(1,(vertex_number - 2)*(vertex_number - 1)/2)
        else:
            edges_number = random.randint(int((int(edge_density) - 1)*((vertex_number) * (vertex_number + 1) / 2)/4),
                                          int(int(edge_density)*((vertex_number) * (vertex_number + 1) / 2)/4))
        for i in range(vertex_number):
            for j in range(i + 1,vertex_number):
                edge = 'vertex'+str(i+1)+',vertex'+str(j+1)
                if is_weighted:
                    edge += ','+str(random.randint(1,10))
                edges_pool+=[edge]
        for k in range(edges_number):
            number = random.randint(0,len(edges_pool) - 1)
            adj_list += [edges_pool.pop(number)]
    else:
        edges_pool = []
        if len(args) < 2:
            edges_number = random.randint(1, (vertex_number - 2) * (vertex_number - 1) / 2)
        else:
            edges_number = random.randint(int((int(edge_density) - 1) * ((vertex_number + 1) * (vertex_number - 1)) / 4),
                                          int(int(edge_density) * ((vertex_number + 1) * (vertex_number)) / 4))
        for i in range(vertex_number):
            for j in range(vertex_number):
                if i != j:
                    edge = 'vertex' + str(i + 1) + ',vertex' + str(j + 1)
                    if is_weighted:
                        edge += ',' + str(random.randint(1, 10))
                    edges_pool += [edge]
        for k in range(edges_number):
            number = random.randint(0, len(edges_pool) - 1)
            adj_list += [edges_pool.pop(number)]

    return vertex_names,vertex_number,is_weighted,is_directed,adj_list



def list_to_matrix(vertex_names,adj_list,is_directed):
    vertex_number = len(vertex_names)
    adj_matrix = np.zeros((vertex_number, vertex_number))
    for i in range(vertex_number):
        for j in range(vertex_number):
            if i != j:
                adj_matrix[i][j] = float('inf')
            else:
                adj_matrix[i][j] = 0

    for i in adj_list:
        edge = i.split(',')
        if edge[0] in vertex_names and edge[1] in vertex_names:
            if len(edge) == 2:
                adj_matrix[vertex_names.index(edge[0])][vertex_names.index(edge[1])] = 1
                if not is_directed:
                    adj_matrix[vertex_names.index(edge[1])][vertex_names.index(edge[0])] = 1
            else:
                adj_matrix[vertex_names.index(edge[0])][vertex_names.index(edge[1])] = edge[2]
                if not is_directed:
                    adj_matrix[vertex_names.index(edge[1])][vertex_names.index(edge[0])] = edge[2]
    return adj_matrix



def matrix_to_list(vertex_names,adj_matrix,is_weighted,is_directed):
    temp_adj_list = []
    for i in range(len(vertex_names)):
        for j in range(len(vertex_names)):
            if adj_matrix[i][j] != float('inf') and adj_matrix[i][j] != 0:
                edge = vertex_names[i]+','+vertex_names[j]
                if is_weighted:
                    edge += ','+str(int(adj_matrix[i][j]))
                if not edge in temp_adj_list:
                    if is_directed:
                        temp_adj_list += [edge]
                    else:
                        if is_weighted  and str(edge.split(',')[1]+','+edge.split(',')[0]+','+edge.split(',')[2]) not in temp_adj_list:
                            temp_adj_list += [edge]
                        if not is_weighted and str(edge.split(',')[1] + ',' + edge.split(',')[0]) not in temp_adj_list:
                            temp_adj_list += [edge]
    return temp_adj_list


def weight_check(vertex_names,adj_matrix):
    is_weighted = False
    for i in range(len(vertex_names)):
        for j in range(len(vertex_names)):
            if adj_matrix[i][j]!=float('inf') and adj_matrix[i][j]!=0 and adj_matrix[i][j]!=1:
                is_weighted = True
    return is_weighted



def make_chain(vertex_set,vertex_names,adj_matrix,need_to_close):
    chain = []
    is_weighted = weight_check(vertex_names,adj_matrix)
    for i in range(len(vertex_set) - 1):
        current_edge = vertex_set[i] + ',' + vertex_set[i + 1]
        if is_weighted:
            current_edge += ',' + str(int(adj_matrix[vertex_names.index(vertex_set[i])][vertex_names.index(vertex_set[i + 1])]))
        chain += [current_edge]
    if need_to_close:
        if not is_weighted:
            chain += [vertex_set[len(vertex_set)-1]+','+vertex_set[0]]
        else:
            chain += [vertex_set[len(vertex_set)-1]+','+vertex_set[0]+','+
                      str(int(adj_matrix[vertex_names.index(vertex_set[len(vertex_set)-1])][vertex_names.index(vertex_set[0])]))]
    return chain


def show_graph(vertex_names,adj_list,is_directed,*args):
    #0 pos in args - edges wanted to be highlighted; input like this: ['a,b,2','b,c,5'] - if graph weighted input weights
    #1 pos to color edges
    if len(args) > 0 and args[0] != None:
        highlighted_edges = args[0]
    if len(args) > 1:
        colored_vertexes = args[1]
        color_pool = []
        for i in range(len(colored_vertexes)):
            color_pool += [(random.random(),random.random(),random.random())]
        dict_colored = {}
        for i in colored_vertexes:
            c = random.choice(color_pool)
            dict_colored[i] = c
            color_pool.remove(c)

    vertex_number = len(vertex_names)
    coordinates = np.array([[np.cos(2*t*np.pi/vertex_number),np.sin(2*t*np.pi/vertex_number)] for t in range(vertex_number)])
    for i in vertex_names:
        x = coordinates[vertex_names.index(i)][0]
        y = coordinates[vertex_names.index(i)][1]
        if len(args) <= 1:
            plt.scatter(x,y,color='red')
        if len(args) > 1 and args[1] != None:
            for j in colored_vertexes:
                if i in colored_vertexes[j]:
                    color_num = j
                    break
            current_color = dict_colored[color_num]
            plt.scatter(x, y, color=current_color)
        plt.text(x,y,i,fontsize='15',color='blue')

    for j in adj_list:
        edge = j.split(',')
        x1 = coordinates[vertex_names.index(edge[0])][0]
        y1 = coordinates[vertex_names.index(edge[0])][1]
        x2 = coordinates[vertex_names.index(edge[1])][0]
        y2 = coordinates[vertex_names.index(edge[1])][1]
        if not is_directed:
            if len(args) == 0 or (len(args)>0 and args[0] == None):
                plt.plot([x1,x2],[y1,y2],color='black')
            elif args[0] != None:
                if is_weighted:
                    if j in highlighted_edges or str(edge[1]+','+edge[0]+','+edge[2]) in highlighted_edges:
                        plt.plot([x1, x2], [y1, y2], color='green')
                    else:
                        plt.plot([x1, x2], [y1, y2], color='black')
                else:
                    if j in highlighted_edges or str(edge[1]+','+edge[0]) in highlighted_edges:
                        plt.plot([x1, x2], [y1, y2], color='green')
                    else:
                        plt.plot([x1, x2], [y1, y2], color='black')
        else:
            if len(args) == 0 or (len(args)>0 and args[0] == None):
                plt.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops={'arrowstyle': '-|>'})
            elif args[0] != None:
                if j in highlighted_edges:
                    plt.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops={'arrowstyle': 'fancy'})
                else:
                    plt.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops={'arrowstyle': '-|>'})
        if len(edge) == 3:
            plt.text((x1 + x2)/2,(y1 + y2)/2,edge[2],color='red')

    if len(args)!=0 and args[0] != None:
        plt.title('From '+highlighted_edges[0].split(',')[0]+' to '+highlighted_edges[len(highlighted_edges)-1].split(',')[1])

    plt.show()


#poisk v glubinu
def DFS(vertex_names,adj_matrix):
    counter = 0
    visited = [False for a in range(len(vertex_names))]
    previsit = [0 for b in range(len(vertex_names))]
    postvisit = [0 for c in range(len(vertex_names))]

    for j in vertex_names:
        if not visited[vertex_names.index(j)]:
            start_vertex,vertex_names,adj_matrix,visited,previsit,postvisit,counter = \
                explore(j,vertex_names,adj_matrix,visited,previsit,postvisit,counter)
            counter += 1

    return previsit,postvisit


def explore(start_vertex,vertex_names,adj_matrix,*args):
    if len(args) == 0:
        counter = 0
        visited = [False for a in range(len(vertex_names))]
        previsit = [0 for b in range(len(vertex_names))]
        postvisit = [0 for c in range(len(vertex_names))]
    else:
        visited = args[0]
        previsit = args[1]
        postvisit = args[2]
        counter = args[3]
    safe_start_vertex = start_vertex
    previsit[vertex_names.index(safe_start_vertex)] = counter
    visited[vertex_names.index(safe_start_vertex)] = True

    for i in vertex_names:
        if not visited[vertex_names.index(i)] and adj_matrix[vertex_names.index(safe_start_vertex)][vertex_names.index(i)] != float('inf'):
            counter+=1
            start_vertex,vertex_names,adj_matrix,visited,previsit,postvisit,counter = \
                explore(i,vertex_names,adj_matrix,visited,previsit,postvisit,counter)

    counter+=1
    postvisit[vertex_names.index(safe_start_vertex)] = counter

    return safe_start_vertex,vertex_names,adj_matrix,visited,previsit,postvisit,counter


#all shortest paths
def FWA(vertex_names,adj_matrix):
    distances = adj_matrix.copy()
    for k in vertex_names:
        for i in vertex_names:
            for j in vertex_names:
                if i!=k and j!=k and i!=j:
                    distances[vertex_names.index(i)][vertex_names.index(j)] = min(distances[vertex_names.index(i)][vertex_names.index(j)],
                        distances[vertex_names.index(i)][vertex_names.index(k)] + distances[vertex_names.index(k)][vertex_names.index(j)])
    return distances

#shortest paths for one vertex
def BFA(start_vertex,vertex_names,adj_list,is_directed):
    distances = [float('inf') for a in range(len(vertex_names))]
    prev = [None for b in range(len(vertex_names))]
    distances[vertex_names.index(start_vertex)] = 0
    prev[vertex_names.index(start_vertex)] = start_vertex
    for n in range(len(vertex_names)-1):
        for e in adj_list:
            edge = e.split(',')
            edge_len = 1
            if len(edge) > 2:
                edge_len = int(edge[2])
            if distances[vertex_names.index(edge[1])] > distances[vertex_names.index(edge[0])] + edge_len:
                distances[vertex_names.index(edge[1])] = distances[vertex_names.index(edge[0])] + edge_len
                prev[vertex_names.index(edge[1])] = edge[0]
            if not is_directed:
                if distances[vertex_names.index(edge[0])] > distances[vertex_names.index(edge[1])] + edge_len:
                    distances[vertex_names.index(edge[0])] = distances[vertex_names.index(edge[1])] + edge_len
                    prev[vertex_names.index(edge[0])] = edge[1]

    #check for negative cycles
    spec_distances = distances.copy()
    for e in adj_list:
        edge = e.split(',')
        edge_len = 1
        if len(edge) > 2:
            edge_len = int(edge[2])
        if spec_distances[vertex_names.index(edge[1])] > spec_distances[vertex_names.index(edge[0])] + edge_len:
            spec_distances[vertex_names.index(edge[1])] = spec_distances[vertex_names.index(edge[0])] + edge_len

    if spec_distances!=distances:
        print('Negative cycle found')
        return


    return distances,prev


#find your way to exit
def SSP(start_vertex,finish_vertex,vertex_names,adj_list,adj_matrix,is_weighted,is_directed):
    args = explore(start_vertex,vertex_names,adj_matrix)
    if not args[3][vertex_names.index(finish_vertex)]:
        print("Can't reach ",finish_vertex)
        return
    distances,prev = BFA(start_vertex,vertex_names,adj_list,is_directed)
    da_way_rev = []
    current_vortex = finish_vertex
    while current_vortex!=start_vertex:
        current_vortex = prev[vertex_names.index(current_vortex)]
        da_way_rev += [current_vortex]
    da_way = list(reversed(da_way_rev)) + [finish_vertex]

    edges_to_highlight = []
    for i in range(len(da_way) - 1):
        current_edge = da_way[i]+','+da_way[i+1]
        if is_weighted:
            current_edge += ','+str(int(adj_matrix[vertex_names.index(da_way[i])][vertex_names.index(da_way[i+1])]))
        edges_to_highlight += [current_edge]

    show_graph(vertex_names,adj_list,is_directed,edges_to_highlight)

    return distances[vertex_names.index(finish_vertex)],da_way


#greedy
def vertex_cover_edges(vertex_names,adj_list,adj_matrix,is_directed):
    matrix_to_mod = adj_matrix.copy()
    chosen_vertex = []
    vertex_degrees = [0 for t in range(len(vertex_names))]
    for i in range(len(vertex_names)):
        good_edges = 0
        for m in range(len(vertex_names)):
            if matrix_to_mod[i][m] != 0 and matrix_to_mod[i][m] != float('inf'):
                good_edges+=1
        vertex_degrees[i] = good_edges

    edjes_to_cover = len(adj_list)
    while edjes_to_cover > 0:
        vertex = vertex_names[vertex_degrees.index(max(vertex_degrees))]
        chosen_vertex += [vertex]
        good_edges = 0
        for j in range(len(vertex_names)):
            if matrix_to_mod[vertex_names.index(vertex)][j]!=float('inf') and matrix_to_mod[vertex_names.index(vertex)][j]!=0:
                good_edges += 1
                if not is_directed:
                    matrix_to_mod[j][vertex_names.index(vertex)] = float('inf')
        edjes_to_cover -= good_edges
        matrix_to_mod[vertex_names.index(vertex)] = [float('inf') for k in range(len(vertex_names))]
        for i in range(len(vertex_names)):
            good_edges = 0
            for m in range(len(vertex_names)):
                if matrix_to_mod[i][m] != 0 and matrix_to_mod[i][m] != float('inf'):
                    good_edges += 1
            vertex_degrees[i] = good_edges

    return chosen_vertex


#max independent vertex set; greedy one
def MIVS(vertex_names,adj_matrix):
    matrix_to_mod = adj_matrix.copy()
    chosen_vertex = []
    vertex_degrees = [0 for t in range(len(vertex_names))]
    for i in range(len(vertex_names)):
        good_edges = 0
        for m in range(len(vertex_names)):
            if matrix_to_mod[i][m] != 0 and matrix_to_mod[i][m] != float('inf'):
                good_edges+=1
        vertex_degrees[i] = good_edges

    vertex_pool = vertex_names.copy()
    while len(vertex_pool)>0:
        min_degree = float('inf')
        zero_degree_vertex = ''
        for q in range(len(vertex_names)):
            if vertex_degrees[q]<min_degree and vertex_degrees[q]!=0:
                min_degree = vertex_degrees[q]
        if min_degree==float('inf'):
            min_degree = 0
            for w in range(len(vertex_names)):
                if vertex_names[w] in vertex_pool and vertex_degrees[w] == 0:
                    zero_degree_vertex = vertex_names[w]
                    break
        if zero_degree_vertex != '':
            vertex =zero_degree_vertex
        else:
            vertex = vertex_names[vertex_degrees.index(min_degree)]
        chosen_vertex += [vertex]
        for j in range(len(vertex_names)):
            if matrix_to_mod[vertex_names.index(vertex)][j]!=float('inf') and matrix_to_mod[vertex_names.index(vertex)][j]!=0:
                matrix_to_mod[j] = [float('inf') for q in range(len(vertex_names))]
                matrix_to_mod[:,j] = [float('inf') for q in range(len(vertex_names))]
                vertex_pool.remove(vertex_names[j])

        vertex_pool.remove(vertex)
        matrix_to_mod[vertex_names.index(vertex)] = [float('inf') for k in range(len(vertex_names))]

        for i in range(len(vertex_names)):
            good_edges = 0
            for m in range(len(vertex_names)):
                if matrix_to_mod[i][m] != 0 and matrix_to_mod[i][m] != float('inf'):
                    good_edges += 1
            vertex_degrees[i] = good_edges

    return chosen_vertex



def Euler_cycle(vertex_names,adj_matrix):
    #проверка
    for i in range(len(vertex_names)):
        edges = list(filter(lambda x: x!= 0 and x!= float('inf'),adj_matrix[i]))
        if len(edges)%2 != 0:
            return 'No Euler cycle'
    matrix_to_mod = adj_matrix.copy()
    vertex_pool = []
    path = []
    vertex = random.choice(vertex_names)
    vertex_pool += [vertex]
    while len(vertex_pool) > 0:
        vertex = vertex_pool[0]
        if len(list(filter(lambda x: x!= 0 and x!= float('inf'),matrix_to_mod[vertex_names.index(vertex)]))) == 0:
            path +=[vertex]
            vertex_pool.remove(vertex)
        else:
            for i in range(len(vertex_names)):
                if matrix_to_mod[vertex_names.index(vertex)][i] != 0 and matrix_to_mod[vertex_names.index(vertex)][i] != float('inf'):
                    vertex1 = vertex_names[i]
                    break
            vertex_pool = [vertex1] + vertex_pool
            matrix_to_mod[vertex_names.index(vertex)][vertex_names.index(vertex1)] = float('inf')
            matrix_to_mod[vertex_names.index(vertex1)][vertex_names.index(vertex)] = float('inf')

    path.pop(len(path) - 1)
    show_graph(vertex_names,matrix_to_list(vertex_names,adj_matrix,weight_check(vertex_names,adj_matrix),False),False,make_chain(path,vertex_names,adj_matrix,True))
    return path


#full search
def Hamilton_cycle(vertex_names,adj_matrix):
    if len(vertex_names) > 10:
        print('Do it on your own risk(hardly calculated for 11 vertexes). Continue?(y/n)')
        keep_workin = input()
        if keep_workin == n or keep_workin == no or keep_workin == No:
            return 'Not calculating'
    all_ways = list(permutations(vertex_names,len(vertex_names)))
    for i in all_ways:
        way_clear = True
        for j in range(len(i) - 1):
            if adj_matrix[vertex_names.index(i[j])][vertex_names.index(i[j+1])] == float('inf') or adj_matrix[vertex_names.index(i[j])][vertex_names.index(i[j+1])] == 0:
                way_clear = False
        if adj_matrix[vertex_names.index(i[len(i) - 1])][vertex_names.index(i[0])] == float('inf') or adj_matrix[vertex_names.index(i[len(i) - 1])][vertex_names.index(i[0])] == 0:
            way_clear = False
        if way_clear:
            show_graph(vertex_names, matrix_to_list(vertex_names,adj_matrix,weight_check(vertex_names,adj_matrix),False), False, make_chain(i, vertex_names, adj_matrix,True))
            return list(i)
    return 'No Hamilton cycle'


#greedy one based on another greedy one
def paint_it(vertex_names,adj_matrix):
    names = vertex_names.copy()
    matrix_to_mod = adj_matrix.copy()
    colors = 0
    dict = {}
    while len(names) > 0:
        colors += 1
        current_set = MIVS(names,matrix_to_mod)
        #maybe there is a nice way to delete some columns&rows idk
        mod_matrix = np.zeros((len(names)-len(current_set),len(names)-len(current_set)))
        i_m = 0
        for i in range(len(names)):
            if not names[i] in current_set:
                j_m = 0
                for j in range(len(names)):
                    if not names[j] in current_set:
                        mod_matrix[i_m][j_m] = matrix_to_mod[i][j]
                        j_m += 1
                i_m += 1
        dict['color'+str(colors)] = current_set
        for name in current_set:
            names.remove(name)
        matrix_to_mod = mod_matrix

    show_graph(vertex_names,matrix_to_list(vertex_names,adj_matrix,weight_check(vertex_names,adj_matrix),False),False,None,dict)

    return colors,dict


vertex_names,adj_list,is_weighted,is_directed = get_graph()

adj_matrix = list_to_matrix(vertex_names,adj_list,is_directed)
print(adj_matrix)

previsit,postvisit = DFS(vertex_names,adj_matrix)
print('DFS output')
for k in range(len(vertex_names)):
    print(vertex_names[k], previsit[k], postvisit[k])

print('All distances:')
print(FWA(vertex_names,adj_matrix))

vertex1 = random.choice(vertex_names)
vertex2 = random.choice(vertex_names)
while vertex2 == vertex1:
    vertex2 = random.choice(vertex_names)

print('All distances from '+vertex1+':')
print(BFA(vertex1,vertex_names,adj_list,is_directed))

print('Da way from '+vertex1+' to '+vertex2+':')
print(SSP(vertex1,vertex2,vertex_names,adj_list,adj_matrix,is_weighted,is_directed))

print('Vertex cover edges:')
print(vertex_cover_edges(vertex_names,adj_list,adj_matrix,is_directed))

if not is_directed:
    print('Max independent vertex set:')
    print(MIVS(vertex_names,adj_matrix))
    print('Euler cycle:')
    print(Euler_cycle(vertex_names,adj_matrix))
    print('Hamilton cycle:')
    print(Hamilton_cycle(vertex_names,adj_matrix))
    print('Greedy coloring:')
    print(paint_it(vertex_names,adj_matrix))

show_graph(vertex_names,adj_list,is_directed)

#ver 1.3 (it's working properly now(maybe))

#little input example
#a b c d e f
#n
#a,b,2 a,c,5 b,f,1 f,e,2 c,e,1
#Or just Random :)
#For testing I'm often using 'Random 8 3 random False'
#report bugs
#Your ad could be here
