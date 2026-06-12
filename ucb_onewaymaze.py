import numpy as np
import random
import matplotlib.pyplot as plt 

print("start")

episode = 10000
q_table = np.zeros((10,2))
n_table = np.zeros((10,2))   # 記錄每個 (s,a) 被選次數
results = np.zeros((episode,1))

alpha = 0.9
gamma = 0.5
c = 2               # UCB 探索強度
total_steps = 1     # 全域時間 t (避免 log(0))

for i in range(episode):
    steps = 0
    state = 0
    finish = 0

    while finish != 1:
        if 0 in n_table[state]:  
            action = np.argmin(n_table[state])
        else:
            ucb_values = q_table[state] + \
                c * np.sqrt(np.log(total_steps) / n_table[state])
            action = np.argmax(ucb_values)

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
        
        # Q-learning update
        q_table[state, action] = q_table[state, action] + \
            alpha * (r + gamma*np.max(q_table[next_state]) - q_table[state, action])

        # 更新次數
        n_table[state, action] += 1
        total_steps += 1

        steps += 1
        state = next_state

    results[i] = steps
    print("episode {} steps: {}".format(i, steps))

print(q_table)
print("Ave steps: {}".format(np.average(results)))

plt.figure(figsize=(100,5))
plt.plot(range(episode), results, linestyle='-', color='blue', label='Steps per episode')
plt.axhline(y=np.min(results), color='red', linestyle='--', label='Minimum steps')
plt.xlabel("Episode")
plt.ylabel("Steps to reach goal")
plt.title("Q-learning Steps per Episode")
plt.legend()
plt.grid(True)
plt.show()