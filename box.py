import numpy as np
import random
import matplotlib.pyplot as plt

episode = 1000
epsilon = 0.9
alpha = 0.2
gamma = 0.5
q_table = np.zeros((21,3))
results = np.zeros((episode, 1))

for i in range(episode):
    state = 0
    finish = 0
    step = 0

    while finish != 1:
        if random.random() < epsilon:
            action = random.randint(0,2)
        else:
            action = np.argmax(q_table[state])

        next_state = state + action + 1
        if next_state == 20:
            r = 10
            finish = 1
        elif next_state > 20:
            next_state = 20
            r = -100
            finish = 1
        else:
            r = -1
        
        q_table[state, action] = q_table[state, action] + alpha*(r + gamma*(max(q_table[next_state])) - q_table[state, action])
        step += 1
        state = next_state

    print("episode {} steps: {}".format(i, step))
    epsilon = max(0, epsilon-0.05)
    results[i] = step

print(q_table)
print("ave steps:{}".format(np.average(results)))

test_state = 0
test_finish = 0
test_round = 0
while test_finish != 1:
    print("test round:{}".format(test_round))
    test_round += 1    
    print("add:",(np.argmax(q_table[test_state]) + 1))
    test_state += np.argmax(q_table[test_state]) + 1
    print("current state: {}".format(test_state))

    if test_state >= 20:
        test_finish = 1
        if test_state == 20:
            print("Train success")
        else:
            print("Train Fail")


plt.figure()
plt.plot(results)
plt.xlabel("Episode")
plt.ylabel("Steps to finish")
plt.title("Q-learning Training Progress")
plt.show()
