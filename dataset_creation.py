import gym
import numpy
import csv
import gym_pull
from random import randint
with open('marios.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i_episode in range(1000):
	print "episode",i_episode
        string = 'ppaquette/SuperMarioBros-{}-{}-v0'.format(randint(1, 8), randint(1, 4))
        env = gym.make(string)
        observation = env.reset()
        distance=0
        while True:
            env.render()
            action = env.action_space.sample()
                # 0: [0, 0, 0, 0, 0, 0],  # NOOP
                # 1: [1, 0, 0, 0, 0, 0],  # Up
                # 2: [0, 0, 1, 0, 0, 0],  # Down
                # 3: [0, 1, 0, 0, 0, 0],  # Left
                # 4: [0, 1, 0, 0, 1, 0],  # Left + A
                # 5: [0, 1, 0, 0, 0, 1],  # Left + B
                # 6: [0, 1, 0, 0, 1, 1],  # Left + A + B
                # 7: [0, 0, 0, 1, 0, 0],  # Right
                # 8: [0, 0, 0, 1, 1, 0],  # Right + A
                # 9: [0, 0, 0, 1, 0, 1],  # Right + B
                # 10: [0, 0, 0, 1, 1, 1],  # Right + A + B
                # 11: [0, 0, 0, 0, 1, 0],  # A
                # 12: [0, 0, 0, 0, 0, 1],  # B
                # 13: [0, 0, 0, 0, 1, 1],  # A + B
            old_action = action
            observation, reward, done, info = env.step(action)
            if info['distance'] > distance:
		a=numpy.append(action,reward)
		b=numpy.append(info['score'],info['distance'])
                spamwriter.writerow(numpy.append(a,b))
                distance = info['distance']
            if done:
		break
