import numpy as np
import time

class Network:
    def __init__(self,sizes):
        self.sizes = sizes
        self.number_of_layers = len(sizes)
        self.biases = [np.random.randn(i,1) for i in sizes[1:]]
        self.weights = [np.random.randn(i,j) for i,j in zip(sizes[:-1],sizes[1:])]


    def sigmoid(self,x):
        return 1/(1+np.exp(-x))


    def sigmoid_deriative(self,x):
        return np.exp(x)/((1+np.exp(x))**2)


    def learn(self,X,Y,show_progress=False,epochs=10000,learn_rate=1,mini_batch_size=None):
        time1 = time.time()
        for i in range(epochs):
            if show_progress:
                print("Epoch "+str(i))
            if mini_batch_size != None:
                batch_index = np.random.choice([i for i in range(len(X))], mini_batch_size)
            for x,y in zip(X,Y):
                if mini_batch_size != None:
                    for i in range(np.size(batch_index)):
                        if X[batch_index[i]] == x:
                            self.back_prop(x, y, learn_rate, show_progress)
                            break
                else:
                    self.back_prop(x, y, learn_rate, show_progress)
        print(str(time.time() - time1)+"s")


    def back_prop(self,x,y,learn_rate,show_progress):
        a,z = self.forward_pass(x)
        grad_a = a[len(a)-1] - y
        if show_progress:
            print(grad_a)
        delta = grad_a*self.sigmoid_deriative(z[len(z)-1])
        delta_w = np.reshape(a[len(a) - 2],(len(a[len(a) - 2]),1)) @ delta.T
        delta_b = delta
        self.weights[len(self.weights) - 1] -= learn_rate * delta_w
        self.biases[len(self.biases) - 1] -= delta_b
        for i in range(1,len(self.biases)):
            delta = (self.weights[len(self.weights)-i]@delta)*self.sigmoid_deriative(z[len(z)-i-1])
            q = np.array(a[len(a)-i-2])
            q = np.reshape(q,(len(a[len(a)-i-2]),1))
            delta_w = q@delta.T
            delta_b = delta
            self.weights[len(self.weights)-i-1] -= learn_rate*delta_w
            self.biases[len(self.biases)-i-1] -= delta_b


    def forward_pass(self,start_layer):
        output_activation_layer = [np.array(start_layer)]
        output_sum_layer = [np.array(start_layer)]
        for w,b in zip(self.weights,self.biases):
            v_o = np.array(output_activation_layer[len(output_activation_layer)-1])
            v_o = np.reshape(v_o,(len(output_activation_layer[len(output_activation_layer)-1]),1))
            v_t = w.T@v_o
            v_wo = w.T@v_o+b
            output_activation_layer.append(self.sigmoid(v_wo))
            output_sum_layer.append(v_wo)
        return output_activation_layer,output_sum_layer


    def use(self,X,Y):
        output = self.forward_pass(X)
        print(output[0][len(output[0])-1]," Error is: ",Y-output[0][len(output[0])-1])
