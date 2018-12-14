#!/usr/bin/python

import sys, math


#########################
#####   HMM Class   #####
#########################

class HMM:



	# Constructor
	def __init__(self, state_file='', transition_file='', observation_file=''):
		if (state_file != '') and (transition_file != ''):
			self.fit(state_file, transition_file)
		if (observation_file != ''):
			self.predict(observation_file)



	# Set emission and transition probability parameters
	def fit(self, state_file, transition_file):
		self.numStates = 0
		self.startState = 0
		self.names = []  # Indexed by state
		self.emissions = []  # Indexed by state
		self.transitions = [[]]

		# Read in state emission probabilities
		with open(state_file) as f: lines = f.readlines()
		for i in range(len(lines)):
			line = lines[i]
			parse_line = line.strip().split()
			if (len(parse_line) != 3) and (line.strip() != ''): 
				sys.stderr.write('Error - in file ' + state_file + ', each line should have three sets of information each separated by a tab.\n')
				sys.stderr.write('Offending line: ' + line + '\n')
			self.names.append(parse_line[0])
			alphabet = parse_line[1].split(',')
			probabilities = parse_line[2].split(',')
			if (len(alphabet) != len(probabilities)):
				sys.stderr.write('Error - in file ' + state_file + ', each line should have the same number of comma-separated symbols after the first tab as it has comma-separated probabilities after the second tab.')
				sys.stderr.write('Offending line: ' + line + '\n')
			self.emissions.append({})
			for j in range(len(alphabet)): self.emissions[i][alphabet[j]] = float(probabilities[j])
		self.numStates = len(self.names)

		# Read in transition probabilities
		self.transitions = [[0 for x in range(self.numStates)] for y in range(self.numStates)]
		with open(transition_file) as f: lines = f.readlines()
		if (len(lines) != self.numStates+1):
			sys.stderr.write('Error - in file ' + transition_file + ', since there are ' + str(self.numStates) + ' states, the file should contain ' + str(self.numStates+1) + ' lines but instead it contains ' + str(len(lines)) + ' lines.\n')
		for i in range(1, len(lines)):  # Ignore header line
			parse_line = lines[i].strip().split()
			if (len(parse_line) != self.numStates+1):
				sys.stderr.write('Error - in file ' + transition_file + ', since there are ' + str(self.numStates) + ' states, each line should start with the name of the state followed by a tab followed by a tab-separated list of ' + str(self.numStates) + ' transition probabilities.\n')
				sys.stderr.write('Offending line: ' + lines[i] + '\n')
			for j in range(1, len(parse_line)): self.transitions[i-1][j-1] = float(parse_line[j])



	# Execute Viterbi algorithm based on observation sequence
	def predict(self, observation_file):
		self.observation = ''
		self.output = ''
		self.score = float('-inf')
		self.finalState = -1
		self.table = [[]]
		self.backtrack = [[]]

		# Read in observation sequence
		with open(observation_file) as f: self.observation = f.read().strip()

		self.viterbi()
		self.determineBacktrack()



	def viterbi(self):
		length = len(self.observation)

		# Initialize tables
		self.table = [[float('-inf') for y in range(length)] for x in range(self.numStates)]
		self.backtrack = [[-1 for y in range(length)] for x in range(self.numStates)]

		# First column, start state
		self.table[self.startState][0] = HMM.log(self.getEmission(self.startState, 0))

		# Fill in each table entry
		for j in range(1, length):  # Column
			for i in range(self.numStates):  # Row
				maxScore = float('-inf')
				maxState = -1
				for k in range(self.numStates):
					prob = self.table[k][j-1] + HMM.log(self.transitions[k][i])
					if (prob > maxScore):
						maxScore = prob
						maxState = k
				self.table[i][j] = maxScore + HMM.log(self.getEmission(i, j))
				self.backtrack[i][j] = maxState

		# Optimal score
		self.score = float('-inf')
		for i in range(self.numStates):
			if (self.table[i][length-1] > self.score):
				self.score = self.table[i][length-1]
				self.finalState = i



	def determineBacktrack(self):
		length = len(self.observation)
		output = []
		j = length-1  # Column in backtrack table
		state = self.finalState
		while (state != -1) and (j >= 0):
			output.append(self.names[state])
			state = self.backtrack[state][j]
			j -= 1
		self.output = ''.join(output[::-1])
		sys.stdout.write(self.output + '\n')
		with open('path.txt', 'w') as out_file: out_file.write(self.output + '\n')



	# Custom function to get emission probability that handles exception of no value
	def getEmission(self, i, j):
		if (self.observation[j] not in self.emissions[i]): return 0.0
		return self.emissions[i][self.observation[j]]



	# Custom log function to handle exception of taking log of 0.0
	@staticmethod
	def log(x):
		if (x == 0.0): return float('-inf')
		return math.log(x)



####################
#####   MAIN   #####
####################

# Only execute the following code if the program is invoked as the "main" program
# from the command line rather than as a helper script invoked from another program.
if __name__ == "__main__":
	if len(sys.argv) < 4:
		sys.stdout.write("\nUSAGE: HMM.py <states.txt> <transitions.txt> <observation.txt>" + "\n\n")
		sys.stdout.write("HMM generates a Hidden Markov Model based on the state and transition information in two files specified by the first two command line arguments. An observation sequence is determined from the a file specified by the third command line argument. After the model is built, the Viterbi algorithm along with backtracking is used to determine the optimal state sequence for the given observation sequence. The optimal state sequence is output to the screen as well as to a file named path.txt.\n\n")
		sys.exit(1)
	#h = HMM(sys.argv[1], sys.argv[2], sys.argv[3])
	h = HMM()
	h.fit(sys.argv[1], sys.argv[2])
	h.predict(sys.argv[3])
