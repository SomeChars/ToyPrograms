import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import permutations
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import sys


class Graph():
    def __init__(self,input_type='Console',method_name='',user_settings=''):
        #Basically Console == GUI, I've just tested how's GUI works on my PC, but Console useful at making class instances
        #Available input methods: String Matrix File Random
        #String input: edge1 edge2 isolated_vertex1 etc...
        #Matrix input:
        #0,1,2
        #1,0,3
        #2,3,0
        #File input: filename (info in file should be in String or Matrix format)
        #Random input: vertex_number edge_density_level(1-4) is_weighted is_directed; set any to "random" for a random result for it
        self.input_type = input_type
        self.method_name = method_name
        self.user_settings = user_settings
        self.vertex_names = []
        self.vertex_number = 0
        self.adj_list = []
        self.adj_matrix = []
        self.is_directed = False
        self.is_weighted = False
        if self.input_type == 'GUI' and (self.method_name == '' or  self.user_settings == ''):
            app = QtWidgets.QApplication(sys.argv)
            window = uic.loadUi('gv.ui')
            window.pushButton_1.clicked.connect(lambda:self.set_method_name_string(window))
            window.pushButton_2.clicked.connect(lambda:self.set_method_name_matrix(window))
            window.pushButton_3.clicked.connect(lambda:self.set_method_name_file(window))
            window.pushButton_4.clicked.connect(lambda:self.set_method_name_random(window))
            window.pushButton_5.clicked.connect(lambda:self.get_user_settings(window))
            window.show()
            app.exec_()
        if self.input_type == 'Console' and self.method_name == '':
            print('Type method you want to use')
            self.method_name = str(input())
        if len(self.method_name) == 0:
            print('Empty Input')
        if self.method_name == 'Random':
            if self.input_type == 'Console' and self.user_settings == '':
                self.user_settings = input()
            if self.user_settings != '':
                self.user_settings = ['this line is useless'] + self.user_settings.split()
            else:
                self.user_settings = ['this line is useless']
            self.make_random_graph(self.user_settings)
            self.adj_matrix = self.list_to_matrix(self.vertex_names,self.adj_list,self.is_directed)
        elif self.method_name == 'File':
            if self.input_type == 'Console' and self.user_settings == '':
                self.user_settings = input()
            self.input_from_file(self.user_settings)
            self.adj_matrix = self.list_to_matrix(self.vertex_names,self.adj_list,self.is_directed)
        elif self.method_name == 'Matrix':
            if self.input_type == 'Console':
                print('You can print it by rows or just paste it')
                if self.user_settings == '':
                    self.user_settings = input()
                make_settings = True
                lines_left = 1
                while True:
                    if lines_left == 0:
                        break
                    line = [float(i) for i in input().split(',')]
                    if make_settings:
                        lines_left = len(line) - 1
                        self.adj_matrix = np.array(line)
                        make_settings = False
                    else:
                        self.adj_matrix = np.vstack((self.adj_matrix, line))
                        lines_left -= 1
            elif self.input_type == 'GUI':
                make_settings = True
                lines = self.user_settings.split('\n')
                for line in lines:
                    converted_line = [float(i) for i in line.split(',')]
                    if make_settings:
                        self.adj_matrix = np.array(converted_line)
                        make_settings = False
                    else:
                        self.adj_matrix = np.vstack((self.adj_matrix, converted_line))

            self.vertex_number = len(self.adj_matrix)
            self.vertex_names = ['vertex' + str(j + 1) for j in range(self.vertex_number)]
            self.is_directed = not np.array_equal(self.adj_matrix, self.adj_matrix.T)
            self.is_weighted = self.weight_check(self.vertex_names, self.adj_matrix)
            self.adj_list = self.matrix_to_list(self.vertex_names, self.adj_matrix, self.is_weighted, self.is_directed)
        elif self.method_name == 'String':
            if self.input_type == 'Console':
                if self.user_settings == '':
                    self.user_settings = input().split()
                else:
                    self.user_settings = self.user_settings.split()
            else:
                self.user_settings = self.user_settings.split()
            self.adj_list = []
            self.vertex_names = []
            for word in self.user_settings:
                word_temp = word.split(',')
                if len(word_temp) == 1:
                    if word_temp not in self.vertex_names:
                        self.vertex_names += [word_temp]
                else:
                    if word_temp[0] not in self.vertex_names:
                        self.vertex_names += [word_temp[0]]
                    if word_temp[1] not in self.vertex_names:
                        self.vertex_names += [word_temp[1]]
                    self.adj_list += [word]
            self.vertex_names.sort()
            self.is_weighted = True
            weight_control = self.adj_list[0].split(',')
            if len(weight_control) == 2:
                self.is_weighted = False
            self.adj_matrix = self.list_to_matrix(self.vertex_names,self.adj_list,True)
            self.is_directed = not np.array_equal(self.adj_matrix, self.adj_matrix.T)
            self.vertex_number = len(self.vertex_names)

    #it's wrong actually, just like < and > because it's actually NP, but it might work sometimes
    def __eq__(self, other):
        if self.vertex_number != other.vertex_number or len(self.adj_list) != len(other.adj_list):
            return False
        self_degrees = {}
        other_degrees = {}
        for i in range(len(self.adj_matrix)):
            degree = len(list(filter(lambda x: x != 0 and x != float('inf'), self.adj_matrix[i])))
            if degree in self_degrees:
                self_degrees[degree] += 1
            else:
                self_degrees[degree] = 1
        for j in range(len(other.adj_matrix)):
            degree =  len(list(filter(lambda x: x != 0 and x != float('inf'), other.adj_matrix[j])))
            if degree in other_degrees:
                other_degrees[degree] += 1
            else:
                other_degrees[degree] = 1
        if self_degrees != other_degrees:
            return False
        self_edges_degrees = []
        other_edges_degrees = []
        for edge in self.adj_list:
            self_edges_degrees += [[len(list(filter(lambda x: x != 0 and x != float('inf'), self.adj_matrix[self.vertex_names.index(edge.split(',')[0])]))),
                                   len(list(filter(lambda x: x != 0 and x != float('inf'), self.adj_matrix[self.vertex_names.index(edge.split(',')[1])])))]]
        for edge in other.adj_list:
            other_edges_degrees += [[len(list(filter(lambda x: x != 0 and x != float('inf'), other.adj_matrix[other.vertex_names.index(edge.split(',')[0])]))),
                                   len(list(filter(lambda x: x != 0 and x != float('inf'), other.adj_matrix[other.vertex_names.index(edge.split(',')[1])])))]]
        for i in self_edges_degrees:
            if i not in other_edges_degrees and [i[1],i[0]] not in other_edges_degrees:
                return False
            if i in other_edges_degrees:
                other_edges_degrees.pop(other_edges_degrees.index(i))
            else:
                other_edges_degrees.pop(other_edges_degrees.index([i[1],i[0]]))
            self_edges_degrees.pop(self_edges_degrees.index(i))
        return True


    def __lt__(self, other):
        if self.vertex_number > other.vertex_number or len(self.adj_list) > len(other.adj_list):
            return False
        all_subgraphs = list(permutations(other.vertex_names,self.vertex_number))
        for g in all_subgraphs:
            flag = True
            for i in range(self.vertex_number):
                for j in range(self.vertex_number):
                    if self.adj_matrix[i][j] != 0 and self.adj_matrix[i][j] != float('inf'):
                        number = other.adj_matrix[other.vertex_names.index(g[i])][other.vertex_names.index(g[j])]
                        if number != self.adj_matrix[i][j]:
                            flag = False
            if flag:
                return True,g
        return False,[]

    def __le__(self,other):
        if self == other:
            return True,self.vertex_names
        else:
            return self < other

    def __gt__(self,other):
        return other < self

    def __ge__(self,other):
        return  other <= self


    def is_connected(self):
        if not self.is_directed:
            pre,post = self.DFS()
            return post[pre.index(0)] == len(self.vertex_names)*2 - 1
        else:
            flag = True
            for vertex in self.vertex_names:
                pre,post = self.DFS(vertex)
                if pre[pre.index(0)] != post[pre.index(0)] + len(self.vertex_names) - 1:
                    flag = False
            return flag


    def is_tree(self):
        return ((len(self.vertex_names)==(len(self.adj_list)+1))and(self.is_connected()))


    def vertex_contraction(self,vertex_set):
        i = 1
        while(True):
            cotracted_vertex = 'contracted_vertex'+str(i)
            if cotracted_vertex not in self.vertex_names:
                break
            i+=1
        for vertex in vertex_set:
            for i in range(len(self.adj_list)):
                if  self.adj_list[i].split(',')[0] == vertex:
                    new_edge = cotracted_vertex+','+self.adj_list[i].split(',')[1]
                    if self.is_weighted:
                        new_edge += ','+self.adj_list[i].split(',')[2]
                    self.adj_list[i] = new_edge
                if self.adj_list[i].split(',')[1] == vertex:
                    new_edge = self.adj_list[i].split(',')[0]+','+cotracted_vertex
                    if self.is_weighted:
                        new_edge += ','+self.adj_list[i].split(',')[2]
                    self.adj_list[i] = new_edge
        for edge in self.adj_list:
            if edge.split(',')[0] == edge.split(',')[1]:
                self.adj_list.pop(self.adj_list.index(edge))
        for vertex in vertex_set:
            self.vertex_names.pop(self.vertex_names.index(vertex))
        self.vertex_names += [cotracted_vertex]
        self.adj_matrix = self.list_to_matrix(self.vertex_names,self.adj_list,self.is_directed)
        self.vertex_number = len(self.vertex_names)



    def __add__(self, other):
        #переименовываю все вершины чтобы избежать коллизии
        renamed_vertexes = ['vertex'+str(i+1) for i in range(len(self.vertex_names)+len(other.vertex_names))]
        renamed_self_edges = []
        renamed_other_edges = []
        for edge in self.adj_list:
            if self.is_weighted:
                renamed_self_edges += [renamed_vertexes[self.vertex_names.index(edge.split(',')[0])]+','+
                                       renamed_vertexes[self.vertex_names.index(edge.split(',')[1])]+','+str(edge.split(',')[2])]
            else:
                renamed_self_edges += [renamed_vertexes[self.vertex_names.index(edge.split(',')[0])] + ',' +
                                       renamed_vertexes[self.vertex_names.index(edge.split(',')[1])]]
        for edge in other.adj_list:
            if other.is_weighted:
                renamed_other_edges += [renamed_vertexes[other.vertex_names.index(edge.split(',')[0])+
                    len(self.vertex_names)]+','+renamed_vertexes[other.vertex_names.index(edge.split(',')[1])+
                    len(self.vertex_names)]+','+str(edge.split(',')[2])]
            else:
                renamed_other_edges += [renamed_vertexes[other.vertex_names.index(edge.split(',')[0])+len(self.vertex_names)]+','+
                                       renamed_vertexes[other.vertex_names.index(edge.split(',')[1])+len(self.vertex_names)]]
        output = ''
        for i in renamed_self_edges:
            if i.split(',')[0] in renamed_vertexes:
                renamed_vertexes.pop(renamed_vertexes.index(i.split(',')[0]))
            if i.split(',')[1] in renamed_vertexes:
                renamed_vertexes.pop(renamed_vertexes.index(i.split(',')[1]))
            output += str(i)+' '
        for i in renamed_other_edges:
            if i.split(',')[0] in renamed_vertexes:
                renamed_vertexes.pop(renamed_vertexes.index(i.split(',')[0]))
            if i.split(',')[1] in renamed_vertexes:
                renamed_vertexes.pop(renamed_vertexes.index(i.split(',')[1]))
            output += str(i)+' '
        for i in renamed_vertexes:
            output += str(i)+' '
        return Graph('Console','String',output[:len(output)-1])


    def add_vertex(self,name):
        if name in self.vertex_names:
            i = 1
            while(True):
                temp_name = 'vertex'+str(i)
                if temp_name not in self.vertex_names:
                    self.vertex_names += [temp_name]
                    break
                i += 1
        self.vertex_number += 1
        self.adj_matrix = self.list_to_matrix(self.vertex_names,self.adj_list,self.is_directed)


    #проверьте кто-нибудь, пожалуйста
    def add_edge(self,edge):
        if edge.split(',')[0] not in self.vertex_names or edge.split(',')[1] not in self.vertex_names:
            print('Do you want to add missing vertex(es)?(y/n)')
            answer = input()
            if answer == 'n':
                return
            temp_name1 = ''
            temp_name2 = ''
            if edge.split(',')[0] not in self.vertex_names:
                i = 1
                while (True):
                    temp_name = 'vertex' + str(i)
                    if temp_name not in self.vertex_names:
                        self.vertex_names += [temp_name]
                        break
                    i += 1
                temp_name1 = temp_name
            if edge.split(',')[1] not in self.vertex_names:
                i = 1
                while (True):
                    temp_name = 'vertex' + str(i)
                    if temp_name not in self.vertex_names:
                        self.vertex_names += [temp_name]
                        break
                    i += 1
                temp_name2 = temp_name
            if temp_name1 != '' and temp_name2 != '':
                if len(edge.split(',')) > 2 and not self.is_weighted:
                    print('Do you want to make graph weighted?(y/n)')
                    answer = input()
                    if answer == 'n':
                        return
                    for i in range(len(self.adj_list)):
                        self.adj_list[i] += ',' + str(1)
                    self.adj_list += [temp_name1+','+temp_name2+','+edge.split(',')[2]]
                elif len(edge.split(',')) == 2 and self.is_weighted:
                    self.adj_list += [temp_name1+','+temp_name2+','+str(1)]
                elif len(edge.split(',')) > 2:
                    self.adj_list += [temp_name1 + ',' + temp_name2 + ',' + edge.split(',')[2]]
                else:
                    self.adj_list += [temp_name1 + ',' + temp_name2]
            if temp_name1 != '' and temp_name2 == '':
                if len(edge.split(',')) > 2 and not self.is_weighted:
                    print('Do you want to make graph weighted?(y/n)')
                    answer = input()
                    if answer == 'n':
                        return
                    for i in range(len(self.adj_list)):
                        self.adj_list[i] += ',' + str(1)
                    self.adj_list += [temp_name1+','+edge.split(',')[1]+','+edge.split(',')[2]]
                elif len(edge.split(',')) == 2 and self.is_weighted:
                    self.adj_list += [temp_name1+','+edge.split(',')[1]+','+str(1)]
                elif len(edge.split(',')) > 2:
                    self.adj_list += [temp_name1 + ',' + edge.split(',')[1] + ',' + edge.split(',')[2]]
                else:
                    self.adj_list += [temp_name1 + ',' + edge.split(',')[1]]
            if temp_name1 == '' and temp_name2 != '':
                if len(edge.split(',')) > 2 and not self.is_weighted:
                    print('Do you want to make graph weighted?(y/n)')
                    answer = input()
                    if answer == 'n':
                        return
                    for i in range(len(self.adj_list)):
                        self.adj_list[i] += ',' + str(1)
                    self.adj_list += [edge.split(',')[0]+','+temp_name2+','+edge.split(',')[2]]
                elif len(edge.split(',')) == 2 and self.is_weighted:
                    self.adj_list += [edge.split(',')[0]+','+temp_name2+','+str(1)]
                elif len(edge.split(',')) > 2:
                    self.adj_list += [edge.split(',')[0] + ',' + temp_name2 + ',' + edge.split(',')[2]]
                else:
                    self.adj_list += [edge.split(',')[0] + ',' + temp_name2]
        else:
            for e in self.adj_list:
                if e == edge:
                    print('This edge is already in graph')
                elif (e.split(',')[0] == edge.split(',')[0] and e.split(',')[1] == edge.split(',')[1]) or (
                        e.split(',')[0] == edge.split(',')[1] and e.split(',')[1] == edge.split(',')[0]):
                    if len(e.split(',')) == len(edge.split(',')):
                        print('Do you want to update weight?(y/n)')
                        answer = input()
                        if answer == 'n':
                            return
                        self.adj_list[self.adj_list.index(e)] = edge
                    elif len(e.split(',')) > len(edge.split(',')):
                        self.adj_list[self.adj_list.index(e)] = e.split(',')[0] + ',' + e.split(',')[1] + ',' + str(1)
                    else:
                        print('Do you want to make graph weighted?(y/n)')
                        answer = input()
                        if answer == 'n':
                            return
                        for i in range(len(self.adj_list)):
                            self.adj_list[i] += ',' + str(1)
                            if (self.adj_list[i].split(',')[0] == edge.split(',')[0] and self.adj_list[i].split(',')[1] == edge.split(',')[1]) or (
                                    self.adj_list[i].split(',')[0] == edge.split(',')[1] and self.adj_list[i].split(',')[1] == edge.split(',')[0]):
                                self.adj_list[i][len(self.adj_list[i])-1] = ''
                                self.adj_list[i] += edge.split(',')[2]



    def set_method_name_string(self,window):
        self.method_name = 'String'
    def set_method_name_matrix(self,window):
        self.method_name = 'Matrix'
    def set_method_name_file(self,window):
        self.method_name = 'File'
    def set_method_name_random(self,window):
        self.method_name = 'Random'
    def get_user_settings(self,window):
        self.user_settings = window.textEdit.toPlainText()
        window.close()


    def make_random_graph(self,args):
        vertex_names = ''
        adj_list = []
        vertex_number = random.randint(1, 16)
        if len(args) > 1 and args[1] != 'random':
            vertex_number = int(args[1])
        vertex_names = ['vertex' + str(j + 1) for j in range(vertex_number)]

        is_directed = random.choice([True, False])
        is_weighted = random.choice([True, False])

        if len(args) > 3 and args[3] != 'random':
            is_weighted = False if args[3] == 'False' else True
        if len(args) > 4 and args[4] != 'random':
            is_directed = False if args[4] == 'False' else True
        if len(args) > 2:
            if args[2] == 'random':
                edge_density = random.randint(1, 4)
            else:
                edge_density = args[2]

        if not is_directed:
            edges_pool = []
            if len(args) < 2:
                edges_number = random.randint(1, (vertex_number - 2) * (vertex_number - 1) / 2)
            else:
                edges_number = random.randint(
                    int((int(edge_density) - 1) * ((vertex_number) * (vertex_number + 1) / 2) / 4),
                    int(int(edge_density) * ((vertex_number) * (vertex_number + 1) / 2) / 4))
            for i in range(vertex_number):
                for j in range(i + 1, vertex_number):
                    edge = 'vertex' + str(i + 1) + ',vertex' + str(j + 1)
                    if is_weighted:
                        edge += ',' + str(random.randint(1, 10))
                    edges_pool += [edge]
            for k in range(edges_number):
                number = random.randint(0, len(edges_pool) - 1)
                adj_list += [edges_pool.pop(number)]
        else:
            edges_pool = []
            if len(args) < 2:
                if vertex_number > 2:
                    edges_number = random.randint(0, (vertex_number - 2) * (vertex_number - 1) / 2)
                elif vertex_number == 2:
                    edges_number = random.randint(0,1)
                else:
                    edges_number = 0
            else:
                edges_number = random.randint(
                    int((int(edge_density) - 1) * ((vertex_number + 1) * (vertex_number - 1)) / 4),
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

        self.vertex_names, self.vertex_number, self.is_weighted, self.is_directed, self.adj_list =\
            vertex_names, vertex_number, is_weighted, is_directed, adj_list

    def input_from_file(self,filename):
        file = open(filename, 'r')
        lines = file.readlines()
        if len(lines) == 3 and (
                lines[1].startswith('y') or lines[1].startswith('yes') or lines[1].startswith('n') or lines[
            1].startswith('no')):
            is_directed = False
            vertex_names = lines[0].split()
            adj_list = lines[2].split()
            directed = lines[1]
            if directed == 'y' or directed == 'yes':
                is_directed = True
            is_weighted = True
            weight_control = adj_list[0].split(',')
            if len(weight_control) == 2:
                is_weighted = False
        else:
            adj_matrix = np.loadtxt(filename, delimiter=',')
            is_directed = not np.array_equal(adj_matrix, adj_matrix.T)
            vertex_names = ['vertex' + str(i + 1) for i in range(len(adj_matrix))]
            is_weighted = self.weight_check(vertex_names, adj_matrix)

        self.vertex_names, self.vertex_number, self.is_weighted, self.is_directed, self.adj_list = \
            vertex_names, len(adj_matrix), is_weighted, is_directed, self.matrix_to_list(vertex_names, adj_matrix,
                                                                                       is_weighted, is_directed)


    def list_to_matrix(self,vertex_names,adj_list,is_directed):
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


    def matrix_to_list(self,vertex_names, adj_matrix, is_weighted, is_directed):
        temp_adj_list = []
        for i in range(len(vertex_names)):
            for j in range(len(vertex_names)):
                if adj_matrix[i][j] != float('inf') and adj_matrix[i][j] != 0:
                    edge = vertex_names[i] + ',' + vertex_names[j]
                    if is_weighted:
                        edge += ',' + str(int(adj_matrix[i][j]))
                    if not edge in temp_adj_list:
                        if is_directed:
                            temp_adj_list += [edge]
                        else:
                            if is_weighted and str(
                                    edge.split(',')[1] + ',' + edge.split(',')[0] + ',' + edge.split(',')[
                                        2]) not in temp_adj_list:
                                temp_adj_list += [edge]
                            if not is_weighted and str(
                                    edge.split(',')[1] + ',' + edge.split(',')[0]) not in temp_adj_list:
                                temp_adj_list += [edge]
        return temp_adj_list


    def weight_check(self,vertex_names, adj_matrix):
        is_weighted = False
        for i in range(len(vertex_names)):
            for j in range(len(vertex_names)):
                if adj_matrix[i][j] != float('inf') and adj_matrix[i][j] != 0 and adj_matrix[i][j] != 1:
                    is_weighted = True
        return is_weighted


    def make_chain(self,vertex_set,need_to_close):
        chain = []
        for i in range(len(vertex_set) - 1):
            current_edge = vertex_set[i] + ',' + vertex_set[i + 1]
            if self.is_weighted:
                current_edge += ',' + str(
                    int(self.adj_matrix[self.vertex_names.index(vertex_set[i])][self.vertex_names.index(vertex_set[i + 1])]))
            chain += [current_edge]
        if need_to_close:
            if not self.is_weighted:
                chain += [vertex_set[len(vertex_set) - 1] + ',' + vertex_set[0]]
            else:
                chain += [vertex_set[len(vertex_set) - 1] + ',' + vertex_set[0] + ',' +
                          str(int(self.adj_matrix[self.vertex_names.index(vertex_set[len(vertex_set) - 1])][
                                      self.vertex_names.index(vertex_set[0])]))]
        return chain


    def show_graph(self, *args):
        # 0 pos in args - edges wanted to be highlighted; input like this: ['a,b,2','b,c,5'] - if graph weighted input weights
        # 1 pos to color edges
        if len(args) > 0 and args[0] != None:
            highlighted_edges = args[0]
        if len(args) > 1:
            colored_vertexes = args[1]
            color_pool = []
            for i in range(len(colored_vertexes)):
                color_pool += [(random.random(), random.random(), random.random())]
            dict_colored = {}
            for i in colored_vertexes:
                c = random.choice(color_pool)
                dict_colored[i] = c
                color_pool.remove(c)

        coordinates = np.array([[np.cos(2 * t * np.pi / self.vertex_number), np.sin(2 * t * np.pi / self.vertex_number)] for t in
                                range(self.vertex_number)])
        for i in self.vertex_names:
            x = coordinates[self.vertex_names.index(i)][0]
            y = coordinates[self.vertex_names.index(i)][1]
            if len(args) <= 1:
                plt.scatter(x, y, color='red')
            if len(args) > 1 and args[1] != None:
                for j in colored_vertexes:
                    if i in colored_vertexes[j]:
                        color_num = j
                        break
                current_color = dict_colored[color_num]
                plt.scatter(x, y, color=current_color)
            plt.text(x, y, i, fontsize='15', color='blue')

        for j in self.adj_list:
            edge = j.split(',')
            x1 = coordinates[self.vertex_names.index(edge[0])][0]
            y1 = coordinates[self.vertex_names.index(edge[0])][1]
            x2 = coordinates[self.vertex_names.index(edge[1])][0]
            y2 = coordinates[self.vertex_names.index(edge[1])][1]
            if not self.is_directed:
                if len(args) == 0 or (len(args) > 0 and args[0] == None):
                    plt.plot([x1, x2], [y1, y2], color='black')
                elif args[0] != None:
                    if self.is_weighted:
                        if j in highlighted_edges or str(edge[1] + ',' + edge[0] + ',' + edge[2]) in highlighted_edges:
                            plt.plot([x1, x2], [y1, y2], color='green')
                        else:
                            plt.plot([x1, x2], [y1, y2], color='black')
                    else:
                        if j in highlighted_edges or str(edge[1] + ',' + edge[0]) in highlighted_edges:
                            plt.plot([x1, x2], [y1, y2], color='green')
                        else:
                            plt.plot([x1, x2], [y1, y2], color='black')
            else:
                if len(args) == 0 or (len(args) > 0 and args[0] == None):
                    plt.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops={'arrowstyle': '-|>'})
                elif args[0] != None:
                    if j in highlighted_edges:
                        plt.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops={'arrowstyle': 'fancy'})
                    else:
                        plt.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops={'arrowstyle': '-|>'})
            if len(edge) == 3:
                plt.text((x1 + x2) / 2, (y1 + y2) / 2, edge[2], color='red')

        if len(args) != 0 and args[0] != None:
            plt.title('From ' + highlighted_edges[0].split(',')[0] + ' to ' +
                      highlighted_edges[len(highlighted_edges) - 1].split(',')[1])

        plt.show()


    # depth search
    def DFS(self,root=''):
        j = 0
        counter = 0
        visited = [False for a in range(len(self.vertex_names))]
        previsit = [0 for b in range(len(self.vertex_names))]
        postvisit = [0 for c in range(len(self.vertex_names))]
        vertex_names = self.vertex_names
        adj_matrix = self.adj_matrix
        if root != '':
            start_vertex, vertex_names, adj_matrix, visited, previsit, postvisit, counter = \
                self.explore(j, vertex_names, adj_matrix, visited, previsit, postvisit, counter)
        for j in vertex_names:
            if not visited[vertex_names.index(j)]:
                start_vertex, vertex_names, adj_matrix, visited, previsit, postvisit, counter = \
                    self.explore(j, vertex_names, adj_matrix, visited, previsit, postvisit, counter)
                counter += 1

        return previsit, postvisit


    def explore(self,start_vertex, vertex_names, adj_matrix, *args):
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
            if not visited[vertex_names.index(i)] and adj_matrix[vertex_names.index(safe_start_vertex)][
                vertex_names.index(i)] != float('inf'):
                counter += 1
                start_vertex, vertex_names, adj_matrix, visited, previsit, postvisit, counter = \
                    self.explore(i, vertex_names, adj_matrix, visited, previsit, postvisit, counter)

        counter += 1
        postvisit[vertex_names.index(safe_start_vertex)] = counter

        return safe_start_vertex, vertex_names, adj_matrix, visited, previsit, postvisit, counter


    # all shortest paths
    def FWA(self):
        distances = self.adj_matrix.copy()
        for k in self.vertex_names:
            for i in self.vertex_names:
                for j in self.vertex_names:
                    if i != k and j != k and i != j:
                        distances[self.vertex_names.index(i)][self.vertex_names.index(j)] = min(
                            distances[self.vertex_names.index(i)][self.vertex_names.index(j)],
                            distances[self.vertex_names.index(i)][self.vertex_names.index(k)] + distances[self.vertex_names.index(k)][
                                self.vertex_names.index(j)])
        return distances


    # shortest paths for one vertex
    def BFA(self,start_vertex):
        distances = [float('inf') for a in range(len(self.vertex_names))]
        prev = [None for b in range(len(self.vertex_names))]
        distances[self.vertex_names.index(start_vertex)] = 0
        prev[self.vertex_names.index(start_vertex)] = start_vertex
        for n in range(self.vertex_number - 1):
            for e in self.adj_list:
                edge = e.split(',')
                edge_len = 1
                if len(edge) > 2:
                    edge_len = int(edge[2])
                if distances[self.vertex_names.index(edge[1])] > distances[self.vertex_names.index(edge[0])] + edge_len:
                    distances[self.vertex_names.index(edge[1])] = distances[self.vertex_names.index(edge[0])] + edge_len
                    prev[self.vertex_names.index(edge[1])] = edge[0]
                if not self.is_directed:
                    if distances[self.vertex_names.index(edge[0])] > distances[self.vertex_names.index(edge[1])] + edge_len:
                        distances[self.vertex_names.index(edge[0])] = distances[self.vertex_names.index(edge[1])] + edge_len
                        prev[self.vertex_names.index(edge[0])] = edge[1]

        # check for negative cycles
        spec_distances = distances.copy()
        for e in self.adj_list:
            edge = e.split(',')
            edge_len = 1
            if len(edge) > 2:
                edge_len = int(edge[2])
            if spec_distances[self.vertex_names.index(edge[1])] > spec_distances[self.vertex_names.index(edge[0])] + edge_len:
                spec_distances[self.vertex_names.index(edge[1])] = spec_distances[self.vertex_names.index(edge[0])] + edge_len

        if spec_distances != distances:
            print('Negative cycle found')
            return

        return distances, prev


    # find your way to exit
    def SSP(self,start_vertex, finish_vertex):
        args = self.explore(start_vertex, self.vertex_names, self.adj_matrix)
        if not args[3][self.vertex_names.index(finish_vertex)]:
            print("Can't reach ", finish_vertex)
            return
        distances, prev = self.BFA(start_vertex)
        da_way_rev = []
        current_vortex = finish_vertex
        while current_vortex != start_vertex:
            current_vortex = prev[self.vertex_names.index(current_vortex)]
            da_way_rev += [current_vortex]
        da_way = list(reversed(da_way_rev)) + [finish_vertex]

        edges_to_highlight = []
        for i in range(len(da_way) - 1):
            current_edge = da_way[i] + ',' + da_way[i + 1]
            if self.is_weighted:
                current_edge += ',' + str(
                    int(self.adj_matrix[self.vertex_names.index(da_way[i])][self.vertex_names.index(da_way[i + 1])]))
            edges_to_highlight += [current_edge]

        self.show_graph(edges_to_highlight)

        return distances[self.vertex_names.index(finish_vertex)], da_way


    # greedy
    def vertex_cover_edges(self):
        matrix_to_mod = self.adj_matrix.copy()
        chosen_vertex = []
        vertex_degrees = [0 for t in range(len(self.vertex_names))]
        for i in range(self.vertex_number):
            good_edges = 0
            for m in range(self.vertex_number):
                if matrix_to_mod[i][m] != 0 and matrix_to_mod[i][m] != float('inf'):
                    good_edges += 1
            vertex_degrees[i] = good_edges

        edjes_to_cover = len(self.adj_list)
        while edjes_to_cover > 0:
            vertex = self.vertex_names[vertex_degrees.index(max(vertex_degrees))]
            chosen_vertex += [vertex]
            good_edges = 0
            for j in range(self.vertex_number):
                if matrix_to_mod[self.vertex_names.index(vertex)][j] != float('inf') and \
                        matrix_to_mod[self.vertex_names.index(vertex)][j] != 0:
                    good_edges += 1
                    if not self.is_directed:
                        matrix_to_mod[j][self.vertex_names.index(vertex)] = float('inf')
            edjes_to_cover -= good_edges
            matrix_to_mod[self.vertex_names.index(vertex)] = [float('inf') for k in range(self.vertex_number)]
            for i in range(self.vertex_number):
                good_edges = 0
                for m in range(self.vertex_number):
                    if matrix_to_mod[i][m] != 0 and matrix_to_mod[i][m] != float('inf'):
                        good_edges += 1
                vertex_degrees[i] = good_edges

        return chosen_vertex


    # max independent vertex set; greedy one
    def MIVS(self,vertex_names,adj_matrix):
        matrix_to_mod = adj_matrix.copy()
        chosen_vertex = []
        vertex_degrees = [0 for t in range(len(vertex_names))]
        for i in range(len(vertex_names)):
            good_edges = 0
            for m in range(len(vertex_names)):
                if matrix_to_mod[i][m] != 0 and matrix_to_mod[i][m] != float('inf'):
                    good_edges += 1
            vertex_degrees[i] = good_edges

        vertex_pool = vertex_names.copy()
        while len(vertex_pool) > 0:
            min_degree = float('inf')
            zero_degree_vertex = ''
            for q in range(len(vertex_names)):
                if vertex_degrees[q] < min_degree and vertex_degrees[q] != 0:
                    min_degree = vertex_degrees[q]
            if min_degree == float('inf'):
                min_degree = 0
                for w in range(len(vertex_names)):
                    if vertex_names[w] in vertex_pool and vertex_degrees[w] == 0:
                        zero_degree_vertex = vertex_names[w]
                        break
            if zero_degree_vertex != '':
                vertex = zero_degree_vertex
            else:
                vertex = vertex_names[vertex_degrees.index(min_degree)]
            chosen_vertex += [vertex]
            for j in range(len(vertex_names)):
                if matrix_to_mod[vertex_names.index(vertex)][j] != float('inf') and \
                        matrix_to_mod[vertex_names.index(vertex)][j] != 0:
                    matrix_to_mod[j] = [float('inf') for q in range(len(vertex_names))]
                    matrix_to_mod[:, j] = [float('inf') for q in range(len(vertex_names))]
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


    def euler_cycle(self,show=False):
        # check
        unexp = []
        for v in range(len(self.vertex_names)):
            count = 0
            for i in range(self.vertex_number):
                if self.adj_matrix[v][i] != 0 and self.adj_matrix[v][i] != float('inf'): count += 1
                if self.adj_matrix[i][v] != 0 and self.adj_matrix[i][v] != float('inf'): count -= 1
            if count != 0:
                unexp += [[self.vertex_names[v],count]]
        if len(unexp) != 2: return [],[]
        saving_edge = ''
        if len(unexp) == 2:
            if unexp[0][1] == 1 and unexp[1][1] == -1:
                saving_edge = unexp[1][0] + ',' + unexp[0][0]
            elif unexp[0][1] == -1 and unexp[1][1] == 1:
                saving_edge = unexp[0][0] + ',' + unexp[1][0]
            else: return [],[]
        adj_list_to_mod = self.adj_list.copy()
        if saving_edge != '': adj_list_to_mod += [saving_edge]
        current_path = [self.vertex_names[0]]
        while len(adj_list_to_mod) > 0:
            success = False
            for e in adj_list_to_mod:
                if e.split(',')[0] == current_path[len(current_path) - 1]:
                    success = True
                    current_path += [e.split(',')[1]]
                    adj_list_to_mod.remove(e)
                    break
                if not self.is_directed:
                    if e.split(',')[1] == current_path[len(current_path) - 1]:
                        success = True
                        current_path += [e.split(',')[0]]
                        adj_list_to_mod.remove(e)
                        break
            if not success:
                if len(adj_list_to_mod) > 0:
                    if current_path[0] == current_path[len(current_path) - 1]:
                        current_path = current_path[1:] + [current_path[1]]

        c_index = current_path.index(saving_edge.split(',')[1])
        current_path = current_path[c_index:] + current_path[1:c_index+1]
        if show:
            self.show_graph(self.make_chain(current_path,False))
        return current_path,self.make_chain(current_path,False)


    # full search
    def hamilton_cycle(self,show=False):
        if self.vertex_number > 10:
            print('Do it on your own risk(hardly calculated for 11 vertexes). Continue?(y/n)')
            keep_workin = input()
            if keep_workin == 'n' or keep_workin == 'no' or keep_workin == 'No':
                return 'Not calculating'
        all_ways = list(permutations(self.vertex_names, self.vertex_number))
        for i in all_ways:
            way_clear = True
            for j in range(len(i) - 1):
                if self.adj_matrix[self.vertex_names.index(i[j])][self.vertex_names.index(i[j + 1])] == float('inf') or \
                        self.adj_matrix[self.vertex_names.index(i[j])][self.vertex_names.index(i[j + 1])] == 0:
                    way_clear = False
            if self.adj_matrix[self.vertex_names.index(i[len(i) - 1])][self.vertex_names.index(i[0])] == float('inf') or \
                    self.adj_matrix[self.vertex_names.index(i[len(i) - 1])][self.vertex_names.index(i[0])] == 0:
                way_clear = False
            if way_clear:
                if show:
                    self.show_graph(self.make_chain(i, True))
                return list(i)
        return 'No Hamilton cycle'


    # greedy one based on another greedy one
    def paint_it(self,show=False):
        names = self.vertex_names.copy()
        matrix_to_mod = self.adj_matrix.copy()
        colors = 0
        dict = {}
        while len(names) > 0:
            colors += 1
            current_set = self.MIVS(names, matrix_to_mod)
            # maybe there is a nice way to delete some columns&rows idk
            mod_matrix = np.zeros((len(names) - len(current_set), len(names) - len(current_set)))
            i_m = 0
            for i in range(len(names)):
                if not names[i] in current_set:
                    j_m = 0
                    for j in range(len(names)):
                        if not names[j] in current_set:
                            mod_matrix[i_m][j_m] = matrix_to_mod[i][j]
                            j_m += 1
                    i_m += 1
            dict['color' + str(colors)] = current_set
            for name in current_set:
                names.remove(name)
            matrix_to_mod = mod_matrix

        if show:
            self.show_graph(None, dict)

        return colors, dict


#ver 2.1 (OOP + GUI + some utility features)
#For testing I'm often using 'Console,Random,8,3,random,False'
#report bugs
#Your ad could be here
