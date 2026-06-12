import numpy as np
import random

q_table = np.zeros((22, 3)) #set q table

#parameters
episode = 10000
epsilon = 0.2
alpha = 0.2
gamma = 0.99

#train q table
for i in range(episode):
    state = 0
    finish = 0
    prev_state = -1
    prev_action = -1

    while finish != 1:
        if state % 4 == 0:
            opp_action = random.randint(1,3)
        else:
            opp_action = 4 - state % 4
        state += opp_action

        if state >= 20:
            if state == 20:
                r = -1
            else:
                r = 1
            
            q_table[prev_state, prev_action] += alpha * (r - q_table[prev_state, prev_action])
            break
        
        if random.random() <= epsilon:     #epsilon greedy
            action = random.randint(0,2)
        else:
            action = np.argmax(q_table[state])

        next_state = state + action + 1
        if next_state >= 21:
            r = -1
            finish = 1
        elif next_state == 20:
            r = 1
            finish = 1
        else:
            r = 0
        
        prev_state = state
        prev_action = action

        if finish == 1:
            q_table[state, action] += alpha * (r - q_table[state, action])
        else:
            q_table[state, action] += alpha * (r + gamma*max(q_table[next_state]) - q_table[state, action])
            state = next_state

        
    #epsilon = max(0.01, epsilon * 0.995)        

print(q_table)

for s in range(1,20):
    print(s," : " , np.argmax(q_table[s]) + 1)

print("Test section")
test_state = 0
while 1:
    player_move = int(input("Please input your steps:"))
    test_state += player_move
    print("Current state:{}".format(test_state))
    if test_state >= 20:
        if test_state == 20:
            print("Player wins!")
        else:
            print("Ai wins!")
        break

    ai_move = np.argmax(q_table[test_state]) + 1
    print("Ai choose to move:{}".format(ai_move))
    test_state += ai_move
    print("Current state:{}".format(test_state))
    if test_state > 20:
        print("Player wins!")
        break