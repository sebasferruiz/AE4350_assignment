# Example file showing a circle moving on screen
import pygame
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from IPython import display

plt.ion()

# pygame setup
#screen = pygame.display.set_mode((1280, 720))
#clock = pygame.time.Clock()
running = True
dt = 0
width = 1280 #screen.get_width()
height = 720 #screen.get_height()
goal_pos = pygame.Vector2(width / 2, height / 4)

rx = 525
ry = 450
rh = 10
rw = 150

rx2 = 660
ry2 = 300

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
        self.pos = pygame.Vector2(width / 2, height-20)
        self.vel = pygame.Vector2(0, 0)
        self.acc = [0,0,0,0]
        self.dir = 0


    def update(self, force):
        self.acc = force
        if self.vel.x == 0:
            self.dir = 0
        else:
            self.dir = np.arctan(self.vel.y/self.vel.x)


        self.vel.x = (-2*np.argmax(force[0:2])+1)*max(force[0:2])*9
        self.vel.y = (-2*np.argmax(force[2:4])+1)*max(force[2:4])*9
        self.pos += self.vel

    #def show(self):
    #    pygame.draw.circle(screen, "white", self.pos, 20)

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

def generate_batch(batch_size, t_max = 500):
    activation = nn.Softmax(dim=0)
    batch_actions,batch_states, batch_rewards = [],[],[]
    #Go through each batch and fill it with data
    for b in range(batch_size):
        #Initialize valiables
        states,actions = [],[]
        total_reward = 0
        #s = env.reset() maybe rocket = Rocket()? s is current state
        rocket = Rocket()
        dist = np.sqrt((goal_pos.x-rocket.pos.x)**2+(goal_pos.y-rocket.pos.y)**2)
        norm_x = translate(rocket.pos.x, 0, width, 0, 1)
        norm_y = translate(rocket.pos.y, 0, height, 0, 1)
        s = np.array([norm_x, norm_y, 1/dist, rocket.acc[0], rocket.acc[1], rocket.acc[2], rocket.acc[3]])
        for t in range(t_max):
            #s_v = torch.FloatTensor([s]) #Convert current state to tensor
            s_v = torch.tensor(s, dtype=torch.float)
            act_probs_v = activation(net(s_v))
            act_probs = act_probs_v.data.numpy()
            final_move = [0,0,0,0]
            a = np.random.choice(len(act_probs), p=act_probs)
            final_move[a] = 1

            rocket.update(final_move)
            #rocket.show()
            #new_s, r, done, info = env.step(a) Perfom a "step"
            dist = np.sqrt((goal_pos.x-rocket.pos.x)**2+(goal_pos.y-rocket.pos.y)**2)
            norm_x = translate(rocket.pos.x, 0, width, 0, 1)
            norm_y = translate(rocket.pos.y, 0, height, 0, 1)


            done = False
            new_s = np.array([norm_x, norm_y, 1/dist, rocket.acc[0], rocket.acc[1], rocket.acc[2], rocket.acc[3]])
            r = 1/dist*900
            if rocket.pos.x < 0 or rocket.pos.x > width or rocket.pos.y < 0 or rocket.pos.y > height:
                r = -1
                #done = True

            elif rocket.pos.x > rx and rocket.pos.x < rx + rw and rocket.pos.y > ry and rocket.pos.y < ry + rh:
                pass
                r = -10000
                #done = True

            elif rocket.pos.x > rx2 and rocket.pos.x < rx2 + rw and rocket.pos.y > ry2 and rocket.pos.y < ry2 + rh:
                pass
                r = -10000
                #done = True

            elif t == t_max-1:
                done = True

            elif dist >= 400:
                r = -10
            elif dist <= 50:
                r = 100

            #record sessions like you did before
            states.append(s)
            actions.append(a)
            total_reward += r

            s = new_s
            if done:
                batch_actions.append(actions)
                batch_states.append(states)
                batch_rewards.append(total_reward)
                break
    return batch_states, batch_actions, batch_rewards



def filter_batch(states_batch,actions_batch,rewards_batch,percentile=50):
    
    reward_threshold = np.percentile(rewards_batch, percentile)


    elite_states = []
    elite_actions = []
    

    for i in range(len(rewards_batch)):
        if rewards_batch[i] >= reward_threshold:

            for j in range(len(states_batch[i])):
                elite_states.append(states_batch[i][j])
                elite_actions.append(actions_batch[i][j])
    
    return elite_states,elite_actions


batch_size = 300
session_size = 300
percentile = 50
hidden_size = 200
learning_rate = 0.01
completion_score = 1000

n_states = 7
n_actions = 4

#neural network
net = Net(n_states, hidden_size, n_actions)
#loss function
objective = nn.CrossEntropyLoss()
#optimisation function
optimizer = optim.Adam(params=net.parameters(), lr=learning_rate)


#########################
#########################
rewards = []
losses = []
for i in range(session_size):
    #generate new sessions
    batch_states,batch_actions,batch_rewards = generate_batch(batch_size, t_max=400)
    elite_states, elite_actions = filter_batch(batch_states,batch_actions,batch_rewards,percentile)
    optimizer.zero_grad()
    #tensor_states = torch.FloatTensor(elite_states)
    #tensor_actions = torch.LongTensor(elite_actions)

    tensor_states = torch.tensor(elite_states, dtype=torch.float)
    tensor_actions = torch.tensor(elite_actions, dtype=torch.long)
    action_scores_v = net(tensor_states)
    loss_v = objective(action_scores_v, tensor_actions)
    loss_v.backward()
    optimizer.step()

    #show results
    mean_reward = np.mean(batch_rewards),
    np.percentile(batch_rewards, percentile)
    print("loss= ",loss_v.item(), "reward_mean= ", mean_reward, "session: ", i)
        
    rewards.append(mean_reward)
    losses.append(loss_v.item())


plt.figure(0)
plt.xlabel("Session")
plt.ylabel("Average reward")
plt.plot(rewards)
plt.tight_layout()
plt.savefig("rewards", dpi=400)

plt.figure(1)
plt.xlabel("Session")
plt.ylabel("Loss")
plt.plot(losses)
plt.tight_layout()
plt.savefig("losses", dpi=400)

torch.save(net.state_dict(), 'best.pth')

#########################
#########################