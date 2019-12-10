import random
import keyboard
import numpy as np
import time
import MyFirstNeuralNetwork
from itertools import groupby


class Map():
    def __init__(self,h_size=10,w_size=10):
        self.h_size = h_size+2
        self.w_size = w_size+2
        self.snake_is_dead = False
        self.score = 0
        self.steps_made = 0
        self.map = [[0 for i in range(self.w_size)] for j in range(self.h_size)]
        for i  in range(self.h_size):
            for j in range(self.w_size):
                if i==0 or i==self.h_size-1 or j==0 or j==self.w_size-1:
                    self.map[i][j]=-1

        #(x,y)
        self.snake_head_pos = [random.randint(int(self.h_size/3),int(self.h_size*2/3)),random.randint(int(self.w_size/3),int(self.w_size*2/3))]
        #it's easy to make it (1,0) (-1,0) (0,1) (0,-1) & just add it to head pos
        self.snake_head_direction = random.choice([[1,0], [-1,0], [0,1], [0,-1]])
        self.snake = []
        self.snake += [self.snake_head_pos]
        self.map[self.snake_head_pos[0]][self.snake_head_pos[1]] = -1


        #add 1st food
        x, y = self.snake_head_pos[0], self.snake_head_pos[1]
        while ([x, y] in self.snake):
            x = random.randint(1, self.h_size-2)
            y = random.randint(1, self.w_size-2)
        self.map[x][y] = 1
        self.food_pos = [x,y]


    def move(self):
        self.steps_made += 1
        snake_head_copy = self.snake_head_pos.copy()
        self.snake_head_pos[0] += self.snake_head_direction[0]
        self.snake_head_pos[1] += self.snake_head_direction[1]
        if self.map[self.snake_head_pos[0]][self.snake_head_pos[1]] == -1:
            self.snake_is_dead = True
            return

        if snake_head_copy not in self.snake:
            self.snake.insert(1,snake_head_copy)

        #eat&draw
        if self.map[self.snake_head_pos[0]][self.snake_head_pos[1]] == 1:
            self.score += 1
            if self.score >= (self.h_size-2)*(self.w_size - 2)-1:
                return True
            x,y =self.snake_head_pos[0],self.snake_head_pos[1]
            while([x,y] in self.snake):
                x,y = random.randint(1, self.h_size-2), random.randint(1, self.w_size-2)
            self.map[x][y] = 1
            self.food_pos = [x, y]
        else:
            c = self.snake[len(self.snake)-1][0],self.snake[len(self.snake)-1][1]
            self.map[self.snake[len(self.snake)-1][0]][self.snake[len(self.snake)-1][1]] = 0
            self.snake = self.snake[:len(self.snake)-1]

        # draw new head
        self.map[self.snake_head_pos[0]][self.snake_head_pos[1]] = -1


    def change_dir(self,new_dir):
        if (new_dir[0] != 0 and new_dir[0] == - self.snake_head_direction[0]) or (new_dir[1] != 0 and new_dir[1] == - self.snake_head_direction[1]):
            return False
        self.snake_head_direction = new_dir


    def give_stats(self):
        return [self.map,self.snake,self.snake_head_pos,self.snake_head_direction,self.food_pos]


    def give_score(self):
        return self.score,self.steps_made


    def dead(self):
        return self.snake_is_dead


    def give_map(self):
        return self.map


    def give_np_map(self):
        return np.array(self.map)


    def give_size(self):
        return self.h_size,self.w_size


class Game():
    def __init__(self,h_size=10,w_size=10,gamemode='simple',snake_AI=None):
        self.game_map = Map(h_size,w_size)
        self.chosen_direction = random.choice([[1,0], [-1,0], [0,1], [0,-1]])
        print(self.chosen_direction)
        self.gamemode = gamemode
        self.snake_AI = snake_AI
        print(self.game_map.give_np_map())

    def game_start(self):
        X = []
        Y = []
        while not self.game_map.dead() and self.game_map.give_score()[0]<(self.game_map.give_size()[0]-2)*(self.game_map.give_size()[1]-2)-1:
            if self.gamemode != 'demonstrate':
                start = time.time()
                while True:
                    if keyboard.is_pressed('UP'):
                        print('up')
                        self.chosen_direction = [-1, 0]
                        break
                    if keyboard.is_pressed('DOWN'):
                        print('down')
                        self.chosen_direction = [1, 0]
                        break
                    if keyboard.is_pressed('LEFT'):
                        print('left')
                        self.chosen_direction = [0, -1]
                        break
                    if keyboard.is_pressed('RIGHT'):
                        print('right')
                        self.chosen_direction = [0, 1]
                        break
                    if self.gamemode == 'simple':
                        if time.time() - start >= 0.5:
                            break
                    if self.gamemode == 'train':
                        if time.time() - start >= 5:
                            print('no choice')
                            break

                if time.time() - start < 0.5:
                    time.sleep(0.5 - time.time() + start)

                if self.gamemode == 'train':
                    x, y = self.make_train_info()
                    X += [x]
                    Y += [y]

            else:
                X1, Y1 = self.make_train_info()
                decision = self.snake_AI.use([X1])
                i = np.amax(decision)
                for j in range(4):
                    if decision[j] == i:
                        index = j
                print(index)
                if index == 0:
                    self.chosen_direction = [1,0]
                if index == 1:
                    self.chosen_direction = [-1,0]
                if index == 2:
                    self.chosen_direction = [0,1]
                if index == 3:
                    self.chosen_direction = [0,-1]
                time.sleep(1)

            self.game_map.change_dir(self.chosen_direction)
            end_game = self.game_map.move()
            if end_game:
                print('YOU WIN!!!')
                if self.gamemode == 'train':
                    return X, Y
                else:
                    return
            print('',flush=True)
            time.sleep(0.25)
            print('\n')

            print(self.game_map.give_np_map())

        print("Game over! your score is "+str(self.game_map.give_score()[0]))
        if self.gamemode == 'train':
            return X,Y


    def make_train_info(self):
        x = []
        y = self.chosen_direction
        stats = self.game_map.give_stats()
        x += [stats[4][0]]
        x += [stats[4][1]]
        x += [stats[2][0]]
        x += [stats[2][1]]
        if stats[3] == [1,0]:
            x += [1]
            x += [0]
            x += [0]
            x += [0]
        if stats[3] == [-1,0]:
            x += [0]
            x += [1]
            x += [0]
            x += [0]
        if stats[3] == [0,1]:
            x += [0]
            x += [0]
            x += [1]
            x += [0]
        if stats[3] == [0,-1]:
            x += [0]
            x += [0]
            x += [0]
            x += [1]
        for e in stats[0]:
            for i in range(len(e)):
                x += [e[i]]
        return x,y



def train_snake(h_size=6,w_size=6,tries=5,epochs=1000,learn_rate=1,use_train_data=False,save_train_data=False,use_AI_data=False,save_AI_data=False,
                train_data_fname='snake_train_data.txt',AI_data_fname='snake_AI_data.txt'):
    if use_AI_data:
        snake_AI = MyFirstNeuralNetwork.Network(None,AI_data_fname)
        return snake_AI
    if use_train_data:
        array_list = []
        with open(train_data_fname) as f_data:
            for k, g in groupby(f_data, lambda x: x.startswith('#')):
                if not k:
                    array_list.append(np.array([[float(x) for x in d.split()] for d in g if len(d.strip())]))
        X = array_list[0]
        Y = array_list[1]
        snake_AI = MyFirstNeuralNetwork.Network([len(X[0]), 4])
        snake_AI.learn(X, Y, False, epochs,learn_rate)
        if save_AI_data:
            snake_AI.save_model(AI_data_fname)
        return snake_AI
    X = []
    Y = []
    for i in range(tries):
        game_for_training = Game(h_size,w_size, 'train')
        x,y = game_for_training.game_start()
        X += x
        for y1 in y:
            if y1 == [1, 0]:
                y2 = [1, 0, 0, 0]
                Y += [y2]
            if y1 == [-1, 0]:
                y2 = [0, 1, 0, 0]
                Y += [y2]
            if y1 == [0, 1]:
                y2 = [0, 0, 1, 0]
                Y += [y2]
            if y1 == [0, -1]:
                y2 = [0, 0, 0, 1]
                Y += [y2]


    # if not use_AI_data and not use_train_data and not save_train_data and not save_AI_data:
    #     snake_AI = MyFirstNeuralNetwork.Network([len(X[0]), 4])
    #     snake_AI.learn(X, Y, False, epochs)


    # note that train data should be like:
    # x1
    # #
    # x2
    # ...
    # xn
    # #
    # y1
    #etc
    if save_train_data:
        f = open(train_data_fname,'w')
        f.write('#\n')
        for x in X:
            for e in x:
                f.write(str(e)+' ')
            f.write('\n')
        f.write('#\n')
        for y in Y:
            for e in y:
                f.write(str(e) + ' ')
            f.write('\n')
    snake_AI = MyFirstNeuralNetwork.Network([len(X[0]), 4])
    snake_AI.learn(X, Y, False, epochs,learn_rate)
    if save_AI_data:
        snake_AI.save_model(AI_data_fname)
    return snake_AI


# test1 = Game(4,4,'demonstrate',train_snake(h_size=4,w_size=4,tries=5,epochs=10000,save_train_data=True,
#                                            save_AI_data=True,train_data_fname='4x4_train_data.txt',AI_data_fname='4x4_AI_data.txt'))
#test1.game_start()
#4*4size 10000epochs is working a little!

#here is some more serious learning:
# test2 = Game(4,4,'demonstrate',train_snake(h_size=4,w_size=4,use_train_data=True,train_data_fname='4x4_train_data.txt',
#                                            save_AI_data=True,AI_data_fname='4x4_AI_data.txt',epochs=100000,learn_rate=0.05))
# test2.game_start()


#well it's something
# test3 = Game(4,4,'demonstrate',train_snake(h_size=4,w_size=4,use_AI_data=True,AI_data_fname='4x4_AI_data.txt'))
# test3.game_start()
