import numpy as np
import time
import random
from itertools import groupby

class Network:
    def __init__(self,sizes,filename=None):
        if filename == None:
            self.sizes = sizes
            self.number_of_layers = len(sizes)
            self.biases = [np.random.randn(i,1) for i in sizes[1:]]
            self.weights = [np.random.randn(i,j) for i,j in zip(sizes[:-1],sizes[1:])]

        # note that data should income like
        # shape
        # #
        # weights
        # #
        # biases
        else:
            array_list = []
            with open(filename) as f_data:
                for k, g in groupby(f_data, lambda x: x.startswith('#')):
                    if not k:
                        array_list.append(np.array([[float(x) for x in d.split()] for d in g if len(d.strip())]))
            self.sizes = array_list[0]
            self.weights = []
            self.biases = []
            for i in range(np.size(self.sizes)-1):
                self.weights += [array_list[i+1]]
                self.biases += [array_list[i+np.size(self.sizes)]]

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))


    def sigmoid_derivative(self,x):
        return np.exp(x)/((1+np.exp(x))**2)


    def learn(self,X,Y,show_progress=False,epochs=10000,learn_rate=1,mini_batch_size=None):
        time1 = time.time()
        for i in range(epochs):
            if show_progress:
                if i%10 == 0:
                    print("Epoch "+str(i))
            if mini_batch_size != None:
                batch_index = random.sample([j for j in range(len(X))],mini_batch_size)
                for i in range(mini_batch_size):
                    self.back_prop(X[i], [Y[i]], learn_rate, show_progress)
            else:
                self.back_prop(X, Y, learn_rate, show_progress)
        print(str(time.time() - time1)+"s")


    def back_prop(self,x,y,learn_rate,show_progress):
        a,z = self.forward_pass(x)
        grad_a = (a[len(a)-1] - np.array(y).T)
        if show_progress:
            print(grad_a)
        delta = grad_a*self.sigmoid_derivative(z[len(z)-1])
        if len(np.shape(np.array([a[len(a) - 2]]).T)) >= 3:
            delta_w = np.array(a[len(a) - 2]).T @ delta.T
        else:
            delta_w = np.array([a[len(a) - 2]]).T @ delta.T
        delta_b = delta
        self.weights[len(self.weights) - 1] -= learn_rate * delta_w
        self.biases[len(self.biases) - 1] -= np.mean(delta_b)
        for i in range(1,len(self.biases)):
            delta = (self.weights[len(self.weights)-i]@delta)*self.sigmoid_derivative(z[len(z)-i-1])
            q = np.array(a[len(a)-i-2])
            q = np.reshape(q,(len(a[len(a)-i-2]),1))
            delta_w = q@delta.T
            delta_b = delta
            self.weights[len(self.weights)-i-1] -= learn_rate*delta_w
            self.biases[len(self.biases)-i-1] -= delta_b


    def forward_pass(self,start_layer):
        output_activation_layer = [start_layer.copy()]
        output_sum_layer = [start_layer.copy()]
        for w,b in zip(self.weights,self.biases):
            v_o = np.array(output_activation_layer[len(output_activation_layer)-1])
            v_wo = w.T @ v_o.T + b
            output_activation_layer.append(self.sigmoid(v_wo))
            output_sum_layer.append(v_wo)
        return output_activation_layer,output_sum_layer


    def save_model(self,filename):
        f = open(filename,'w')
        f.write('#\n')
        for e in self.sizes:
            f.write(str(e)+' ')
        for X in self.weights:
            f.write('\n#\n')
            for x in X:
                for e in x:
                    f.write(str(e) + ' ')
                f.write('\n')
        for X in self.biases:
            f.write('#\n')
            for x in X:
                for e in x:
                    f.write(str(e) + ' ')
                f.write('\n')




    def use(self,X,Y=None):
        output = self.forward_pass(X)
        if Y != None:
            print(output[0][len(output[0])-1]," Error is: ",Y-output[0][len(output[0])-1])
        return output[0][len(output[0])-1]


# X = [[0,0],[0,1],[1,0],[1,1]]
# Y = [0,0,0,1]
# AND = Network([2,1])
# AND.learn(X,Y,False,10000,1,4)
#
# AND.use([1,1],[1])
# AND.use([1,0],[0])
# AND.use([0,1],[0])
# AND.use([0,0],[0])
# AND.save_model('AND_model.txt')
# AND_from_file = Network(None,'AND_model.txt')
# AND_from_file.use([1,1],[1])
# AND_from_file.use([1,0],[0])
# AND_from_file.use([0,1],[0])
# AND_from_file.use([0,0],[0])

# print("--------------")
# X = [[0,0],[0,1],[1,0],[1,1]]
# Y = [0,1,1,0]
# XOR1 = Network([2,2,1])
# XOR1.learn(X,Y,False,10000,0.5)
# XOR1.use([1,1],[0])
# XOR1.use([0,1],[1])
# XOR1.use([1,0],[1])
# XOR1.use([0,0],[0])
# print("--------------")
# X = [[0,0],[0,1],[1,0],[1,1]]
# Y = [0,1,1,0]
# XOR2 = Network([2,2,1])
# XOR2.learn(X,Y,False,10000,1,2)
# XOR2.use([1,1],[0])
# XOR2.use([0,1],[1])
# XOR2.use([1,0],[1])
# XOR2.use([0,0],[0])
# print("--------------")
# X = [[0,0],[0,1],[1,0],[1,1]]
# Y = [0,1,1,1]
# OR = Network([2,1])
# OR.learn(X,Y,False,10000,1,2)
# OR.use([1,1],[1])
# OR.use([0,1],[1])
# OR.use([1,0],[1])
# OR.use([0,0],[0])
# print("--------------")