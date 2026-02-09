import numpy as np
import random
import matplotlib.pyplot as plt 

print("start")
q_table = np.zeros((10,2))
results = np.zeros((100,1))

epsilon = 0.9
alpha = 0.9
gamma = 0.5

for i in range(100):
    steps = 0
    state = 0
    finish = 0

    while finish != 1:
        if random.random() < epsilon:
            action = random.randint(0,1)
        else:
            action = np.argmax(q_table[state])

        if action == 0:
            next_state = state - 1
        if action == 1:
            next_state = state + 1

        if next_state == 9:
            r = 10
            finish = 1
        elif next_state < 0:
            r = -100
            next_state = 0
        else:
            r = -1
        
        q_table[state, action] = q_table[state, action] + alpha*(r + gamma*(max(q_table[next_state])) - q_table[state, action])
        steps += 1
        state = next_state

    epsilon = max(0.05, epsilon-0.05)
    results[i] = steps
    print("episode {} steps: {}".format(i, steps))

print(q_table)
print("Ave steps: {}".format(np.average(results)))

plt.figure(figsize=(10,5))
plt.plot(range(100), results, marker='o', linestyle='-', color='blue', label='Steps per episode')
plt.axhline(y=np.min(results), color='red', linestyle='--', label='Minimum steps')
plt.xlabel("Episode")
plt.ylabel("Steps to reach goal")
plt.title("Q-learning Steps per Episode")
plt.legend()
plt.grid(True)
plt.show()