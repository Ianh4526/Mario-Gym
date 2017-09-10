import gym
import gym_pull
#gym_pull.pull('github.com/ppaquette/gym-super-mario')
env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
for i_episode in range(1):
    observation = env.reset()
    distance = 0
    action=[0, 0, 0, 0, 0, 0]
    while True:
        env.render()
	
        
        observation, reward, done, info = env.step(action)
        #if info['distance'] <= distance:	   
	   #action = [0, 0, 0, 0, 0, 1]	env.action_space.sample()   
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
	#distance = info['distance']
        #[up,down,left,right,a,b]
        while info['distance'] > distance:
		action = [0, 0, 0, 0, 1, 0]
                
        print reward, done, info, action, distance
        distance = info['distance']
       
         #observation, 
        if done:
    		break
