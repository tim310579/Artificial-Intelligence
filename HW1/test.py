"""Test script for the Traffic Jam Puzzle AI for Project 1 of CSCI373 AI.
(c) 2016 Gary Chen
"""
from JamPuzzle import *
from JamAgent import *
from sys import argv
import datetime
import psutil
import os


begin = datetime.datetime.now()

useless, algo, filename = argv
#print(algo, filename)
f = open(filename, 'r')
k = f.readlines()
#for lines in k:
        #print(lines)

tmp = []
for lines in k:
        words = lines.split(' ')
        #print(words[3])
        ori = Orientations.vertical
        length = VehicleTypes.car
        words[4] = words[4][0]
        if words[4] == '1': ori = Orientations.horizontal
        if words[3] == '3': length = VehicleTypes.truck
        tmp.append(Vehicle((int(words[2]), int(words[1])), ori, length, words[0]))
        #print(words[2], words[1], words[4], words[3])


trafficJamtmp = JamPuzzle(6, 6, 2, tmp)
f.close()

#print(trafficJamtmp)
def printSolution(puzzle, solution):
	"""Method that takes an initial puzzle and array of solution Move
	objects and applies the solution to the puzzle while printing out
	the state and next move details for each move.
	"""
	for m in solution:
		print(puzzle)
		print(m)
		puzzle.move(m.pos, m.moves)
	print(puzzle)
	


# Create AI agent and run on specified puzzles
agent = JamAgent()


solution = ''
if algo == '1':
        solution = agent.bfs(trafficJamtmp)
        #print('BFS')
elif algo == '2':
        solution = agent.dfs(trafficJamtmp)
        #print('DFS')
elif algo == '3':
        solution = agent.ids(trafficJamtmp)
        #print('IDS')
elif algo == '4':
        solution = agent.a_star(trafficJamtmp)
        #print('A*')
elif algo == '5':
        solution = agent.ida_star(trafficJamtmp)
        #print('IDA*')
#solution = agent2.dfs(trafficJamtmp)
printSolution(trafficJamtmp, solution)
print('Algorithm: ', end='')
if algo == '1': print('BFS')
elif algo == '2': print('DFS')
elif algo == '3': print('IDS')
elif algo == '4': print('A*')
elif algo == '5': print('IDA*')

print("Puzzle completed in " + str(len(solution)) + " moves.")
print("Number of nodes visited in search:  " + str(agent.nodesVisited))
#print("Space: " + str(agent.space))
end = datetime.datetime.now()
print('Time: ', end-begin)

info = psutil.virtual_memory()


f = open('result/statistic.txt', 'a')
f.write(str(len(solution)) + ' ' + str(agent.nodesVisited) + ' ' + str(end-begin) + '\n')
f.close()
