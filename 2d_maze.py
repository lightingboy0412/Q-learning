import numpy as np
import random

maze = np.array([
    [0,0,0,0,0,1],
    [0,1,1,1,0,1],
    [0,0,0,1,0,0],
    [0,1,0,1,1,1],
    [1,0,0,0,0,1],
    [1,1,0,1,0,1],
    [1,1,0,1,0,2]
])

cols = 6
rows = 7
actions = 4
q_table = np.zeros((rows, cols, actions))
episode = 1000
alpha = 0.2
gamma = 0.5
epsilon = 0.9
max_steps = 30

for i in range(episode):
    r,c = 0,0
    steps = 0
    finish = 0

    while finish != 1 and steps < max_steps:    
        if random.random() < epsilon:
            action = random.randint(0,3)
        else:
            action = np.argmax(q_table[r, c])
        
        if action == 0:
            new_r = r - 1
            new_c = c 
        elif action == 1:
            new_r = r 
            new_c = c - 1
        elif action == 2:
            new_r = r + 1
            new_c = c 
        else:
            new_r = r 
            new_c = c + 1

        bump = 0
        if new_c < 0 or new_r < 0 or new_c >= cols or new_r >= rows:
            reward = -50
            bump = 1
        elif maze[new_r, new_c] == 1:
            reward = -50
            bump = 1
        elif maze[new_r, new_c] == 2:
            reward = 100
            finish = 1
        else:
            reward = -1

        if finish != 1 and bump != 1:
            q_table[r, c, action] = q_table[r, c, action] + alpha * (reward + gamma * max(q_table[new_r, new_c]) - q_table[r, c, action])
        else:
            q_table[r, c, action] = q_table[r, c, action] + alpha * (reward - q_table[r, c, action])

        if bump != 1:
            r = new_r
            c = new_c

        steps += 1
    
    print("episode {} steps:{}".format(i, steps))
    epsilon = max(0.1, epsilon - 0.05)

print(q_table)
print(maze)

test_c , test_r = 0, 0
test_finish = 0
test_steps = 0
test_maze = maze

while test_finish != 1 and test_steps < max_steps:
    print("current (c,r) = ({}, {})".format(test_r, test_c))
    test_maze[test_r, test_c] = 6
    action = np.argmax(q_table[test_r, test_c])
    if action == 0:
        test_r -= 1
    elif action == 1:
        test_c -= 1
    elif action == 2:
        test_r += 1
    else:
        test_c += 1
    
    test_steps += 1

    if maze[test_r, test_c] == 2:
        test_finish = 1

    if test_r < 0 or test_c < 0 or test_r >= rows or test_c >= cols or maze[test_r, test_c] == 1:
        break  

if test_finish == 1:
    print("Train sucessful")
else:
    print("Train fail")

print(test_maze)