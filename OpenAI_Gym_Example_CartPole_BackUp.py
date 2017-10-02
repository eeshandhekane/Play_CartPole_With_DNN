import gym # For exercise!!
import tflearn # High level library on top of tensorflow
from tflearn.layers.core import input_data, fully_connected, dropout # Required layers
from tflearn.layers.estimator import regression # Required algorithm
import numpy as np
from numpy import random # For generating random inputs
from collections import Counter
import time # Just for the sake of time comparisons


# Define parameters
lr = 1e-3 # Learning rate
env = gym.make('CartPole-v0') # Load the cartpole environment
env.reset() # A ritual to roll the environment in
goal = 500 # Number of times we want to balance the pole in order to win
score_threshold = 50 # We want to learn from games with a specific threshold
init_games = 1000 # Do not keep it too large to make it a brute force
play_games = 20 # Define the number of games on which we want to test the neural nets


# Define some random initial games
def GenerateRandomGames():
	for episodes in range(5):
		# For each episode, reset the environment
		env.reset()
		# For time steps in goal range, render the environment to get random games
		for times in range(goal):
			# This just renders the environment (shows screen) with goal number of steps (shows the same thing those many times), for 5 episodes
			env.render() # The rendered image for goal number of iterations is the same
			action = env.action_space.sample() # Generate a random action!
			observation, reward, done, info = env.step(action) # The step takes action as input and returns the observation, reward, done (a boolean) and info.
			# There is a message that says that step is being carried out despite done = 'True' is being returned by the environment
			# Avoid this!!!
			if done:
				break


# Generate a population
def GenerateGamePopulation():
	# If the score is "good", only then retain the animal of the population
	tr_data = []
	scores = []
	accepted_scores = []
	# Loop to generate games populations. Here, we generate 10000 games
	for times in range(init_games):
		# For each new game, define the score to be 0, game memory empty and empty rpevious observation
		score = 0
		game_mem = []
		prev_obs = [] # We will be storing observation from the env.step() function into this variable and it is an array
		# Reset the environment
		env.reset()
		## Print the current game number!!
		##print('Current iteration number : ' + str(times))
		# Get goal steps number of actions on the environment and check the result
		for times2 in range(goal):
			# Define the 0, 1 (left or right) action
			action = random.randint(0, 2) # Generate 0, 1
			# The step takes action as input and returns the observation, reward, done (a boolean) and info.
			## This renders the screen! Use with caution as it eats up huge time!
			## With render, time for 1000 games is 162.5 sec and without it, it is 0.21 sec! 750x speed-up! :P
			#env.render() 
			observation, reward, done, info = env.step(action) 
			# It makes sense to store the previous observation and the current action that gave good reward (Does it really??)
			if len(prev_obs) > 0:
				game_mem.append([prev_obs, action]) # Unnecessary detail to check if prev_obs is not empty (i.e., start to store prev data from second iteration onwards)
			prev_obs = observation
			score += reward
			if done:
				break
		# Check if the whole game was better!
		if score >= score_threshold:
			accepted_scores.append(score)
			# Encode the action as a one-hot vector
			# Append to the training data the previous observation and the action that got the better score!!
			for data in game_mem:
				if data[1] == 1:
					tr_data.append([data[0], [0, 1]])
				elif data[1] == 0:
					tr_data.append([data[0], [1, 0]])		
		# Save the score		
		scores.append(score)
	# Save the training data
	training_data_store = np.array(tr_data)
	np.save('train.npy', training_data_store)
	# Average accepted scores are printed
	print('Average accepted scores : ' + str(float(sum(accepted_scores))/len(accepted_scores)))
	# Also, return the training data
	return tr_data
	# We will train a neural net to fit this training data and then use it to play the game itself!!


# Define a neural network model
def DefineNeuralNetworkModel(input_size):
	# Add input layer
	model = input_data(shape = [None, input_size, 1], name = 'input')
	# Add fully connected layer
	model = fully_connected(model, 128, activation = 'relu')
	model = dropout(model, 0.8) # pkeep = 0.8
	# Add fully connected layer
	model = fully_connected(model, 256, activation = 'relu')
	model = dropout(model, 0.8) # pkeep = 0.8
	# Add fully connected layer
	model = fully_connected(model, 512, activation = 'relu')
	model = dropout(model, 0.8) # pkeep = 0.8
	# Add fully connected layer
	model = fully_connected(model, 256, activation = 'relu')
	model = dropout(model, 0.8) # pkeep = 0.8
	# Add fully connected layer
	model = fully_connected(model, 128, activation = 'relu')
	model = dropout(model, 0.8) # pkeep = 0.8
	# Add output layer
	model = fully_connected(model, 2, activation = 'softmax') # Predict the output action
	# Define the regression problem, optimizer, loss function and learning rate
	model = regression(model, optimizer = 'adam', learning_rate = lr, loss = 'categorical_crossentropy', name = 'outputs')
	# Convert the model to DNN and return
	DNN_model = tflearn.DNN(model)
	return DNN_model


# Define a function to train the model
# This function can be used to train a default model if someone inputs one.
# But if it is not inputted, it will first create a model and then train it.
def TrainNeuralNetworkModel(tr_data, model = False):
	# Define training data input
	X = np.array([ data[0] for data in tr_data ]).reshape(-1, len(tr_data[0][0]), 1)
	y = [ data[1] for data in tr_data ]
	# If model is not inputted, define it
	if not model:
		model = DefineNeuralNetworkModel(input_size = len(X[0]))
	# Fit the model on the training data
	model.fit({ 'input' : X }, { 'outputs' : y }, n_epoch = 100, snapshot_step = 500, show_metric = True, run_id = 'openAIGym' )
	# Save the model
	model.save('DNN_CartPole-v0.model')
	## In case we want to load the model, we need to define the model using the DefineNeuralNetworkModel() script, for which input_size must be known
	## Then, we can use the following script--
	#model = DefineNeuralNetworkModel(input_size)
	#model = model.load('DNN_cartPole-v0.model')
	# Return the trained model
	return model


# Define a function to test the trained model
def PlayCartPoleWithDNN(model):
	# A catalogue of what happened in the test!!
	game_scores = []
	game_actions = []
	# Play game!
	for a_game in range(play_games):
		# Keep a track!
		score = 0
		game_mem = []
		prev_obs = []
		action_list = []
		# Reset the env
		env.reset()
		# For each step,
		for a_step in range(goal):
			# Now, we must see the game, no matter how much time it takes!!
			env.render()
			# Define the actions.
			# If this is the first step, take any action of your choice OR random action
			if len(prev_obs) == 0:
				action = random.randint(0, 2)
			else :
				action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
			# Remember the action!
			action_list.append(action)
			# Make the step
			new_obs, reward, done, info = env.step(action)
			# Set the previous observation value
			prev_obs = new_obs
			# Update the game memory
			game_mem.append([new_obs, action])
			# Increment the score
			score += reward
			# If we fail, stop the game! 
			if done:
				break
		# Remember the score
		game_scores.append(score)
		game_actions.append(action_list)
	# Average score after training
	print('The trained neural net has average score : ' + str(float(sum(game_scores))/len(game_scores)))
	# Return the game scores and game actions
	return game_scores, game_actions

# Define a main
if __name__ == "__main__":
	##################################################
	#GenerateRandomGames()
	##################################################
	##################################################
	#time1 = time.time()
	#GenerateGamePopulation()
	#time2 = time.time()
	#print('Time taken : ' + str(float(time2 - time1)))
	##################################################
	##################################################
	#DefineNeuralNetworkModel(100)
	##################################################
	##################################################
	#tr_data = GenerateGamePopulation()
	#model = TrainNeuralNetworkModel(tr_data)
	##################################################
	##################################################
	train_data = GenerateGamePopulation()
	model = TrainNeuralNetworkModel(train_data)
	PlayCartPoleWithDNN(model)


