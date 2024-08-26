# Example file showing a circle moving on screen
import pygame
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from collections import deque

import matplotlib.pyplot as plt
from IPython import display
import time

plt.ion()

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
dt = 0
lifespan = 1000
count = 0

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
loss = 0
width = screen.get_width()
height = screen.get_height()
goal_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 4)

rx = 525
ry = 450
rh = 10
rw = 150



def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

class Rocket:
    def __init__(self):
        self.pos = pygame.Vector2(screen.get_width() / 2, screen.get_height()-20)
        self.vel = pygame.Vector2(0, 0)
        self.acc = [0,0,0,0]
        self.dir = 0


    def update(self, force):
        self.acc = force
        if self.vel.x == 0:
            self.dir = 0
        else:
            self.dir = np.arctan(self.vel.y/self.vel.x)


        self.vel.x = (-2*np.argmax(force[0:2])+1)*max(force[0:2])
        self.vel.y = (-2*np.argmax(force[2:4])+1)*max(force[2:4])
        #self.vel = clamp_norm(self.vel, 5)
        self.pos += self.vel
        #self.acc = self.acc*0

    def show(self):
        pygame.draw.circle(screen, "white", self.pos, 20)


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

rocket = Rocket()
done = False
plot_scores = []
plot_mean_scores = []
losses = []
total_score = 0
record = 0

model = Net(7, 200, 4)
model.load_state_dict(torch.load('config2.pth'))
activation = nn.Softmax(dim=0)
poses = []
while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


    # fill the screen with a color to wipe away anything from last frame
    screen.fill("black")
    pygame.draw.circle(screen, "red", goal_pos, 20)
    pygame.draw.rect(screen, "white", pygame.Rect(rx, ry, rw, rh))
    #pygame.draw.rect(screen, "white", pygame.Rect(rx2, ry2, rw, rh))

    ###########################
    ###########################
    cur_pos = pygame.Vector2(rocket.pos.x, rocket.pos.y)
    poses.append(cur_pos)
    for pos in poses:
        pygame.draw.circle(screen, "blue", pos, 3)

    dist = np.sqrt((goal_pos.x-rocket.pos.x)**2+(goal_pos.y-rocket.pos.y)**2)
    norm_x = translate(rocket.pos.x, 0, width, 0, 1)
    norm_y = translate(rocket.pos.y, 0, height, 0, 1)
    s = np.array([norm_x, norm_y, 1/dist, rocket.acc[0], rocket.acc[1], rocket.acc[2], rocket.acc[3]])
    s_v = torch.tensor(s, dtype=torch.float)
    act_probs_v = activation(model(s_v))
    act_probs = act_probs_v.data.numpy()
    final_move = [0,0,0,0]
    a = np.random.choice(len(act_probs), p=act_probs)
    final_move[a] = 1

    rocket.update(final_move)
    rocket.show()

    


    if rocket.pos.x > rx and rocket.pos.x < rx + rw and rocket.pos.y > ry and rocket.pos.y < ry + rh:
        print("crash")
    #print(rocket.vel.x, rocket.vel.y, forces[count])

    count += 1
    if count == lifespan:
        count = 0
        rocket = Rocket()
        print('done')
        time.sleep(5)




    #print('Game', agent.n_games, 'Score', score, 'Record:', record)
    #plot(plot_scores, plot_mean_scores)
    ##########################
    ##########################

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000

pygame.quit()