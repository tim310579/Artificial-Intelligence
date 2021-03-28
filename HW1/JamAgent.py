
from JamPuzzle import *
from collections import deque
import copy
from queue import PriorityQueue
class JamAgent:
	
	def __init__(self):
		self.nodesVisited = 0
		self.space = 0

	def bfs(self, puzzle):
		
		bfsQueue = deque([])
		self.nodesVisited = 0
		self.space = 0
		# The current node/state
		current = BfsNode(puzzle, [])
		
		seenPuzzleStates = {}
		seenPuzzleStates[str(current.puzzle.getGrid())] = True;
		
		while not current.puzzle.won():
			self.nodesVisited += 1
			
			for m in current.getPossibleMoves():

				# Duplicate puzzle state and perform a move
				newState = copy.deepcopy(current)
				newState.puzzle.move(m.pos, m.moves)
				self.space += 1
				# If new state is unseen, add to queue and seen states list
				if ((not str(newState.puzzle) in seenPuzzleStates) or seenPuzzleStates[str(newState.puzzle)] > len(newState.movesSoFar)):
					bfsQueue.append(BfsNode(newState.puzzle, current.movesSoFar + [m]))
					seenPuzzleStates[str(newState.puzzle)] = True;
			current = bfsQueue.popleft()
		
		return current.movesSoFar
	def dfs(self, puzzle):
		
		# Queue to hold untraversed nodes
		dfsQueue = deque([])
		self.nodesVisited = 0

		# The current node/state
		current = DfsNode(puzzle, [])
		
		seenPuzzleStates = {}
		seenPuzzleStates[str(current.puzzle.getGrid())] = True;

		while not current.puzzle.won():
			self.nodesVisited += 1
			
			for m in current.getPossibleMoves():

				# Duplicate puzzle state and perform a move
				newState = copy.deepcopy(current)
				newState.puzzle.move(m.pos, m.moves)

				# If new state is unseen, add to queue and seen states list
				if ((not str(newState.puzzle) in seenPuzzleStates) or seenPuzzleStates[str(newState.puzzle)] > len(newState.movesSoFar)):
					dfsQueue.append(DfsNode(newState.puzzle, current.movesSoFar + [m]))
					seenPuzzleStates[str(newState.puzzle)] = True;
			current = dfsQueue.pop()
		
		return current.movesSoFar
	def ids(self, puzzle):
		
		cnt = 0
		idsQueue = PriorityQueue()
		self.nodesVisited = 0
		self.space = 0
		# The current node/state
		current = IdsNode(puzzle, [], 0)
		seenPuzzleStates = {}
		seenPuzzleStates[str(current.puzzle.getGrid())] = True
		#print(type(current))
		while not (current.puzzle.won()):
                        if 1 == 1:
                                self.nodesVisited += 1
			
                                for m in current.getPossibleMoves():
                                        cnt += 1
                                        #print('iiii')
                                        # Duplicate puzzle state and perform a move
                                        newState = copy.deepcopy(current)
                                        newState.puzzle.move(m.pos, m.moves)
                                        self.space += 1
                                        # If new state is unseen, add to queue and seen states list
                                        if ((not str(newState.puzzle) in seenPuzzleStates) or seenPuzzleStates[str(newState.puzzle)] > len(newState.movesSoFar)):
                                                #idsQueue.append(IdsNode(current.thedeep+1, newState.puzzle, current.movesSoFar + [m]))
                                                idsQueue.put((current.thedeep+1, cnt, IdsNode(newState.puzzle, current.movesSoFar + [m], current.thedeep+1)))
                                                seenPuzzleStates[str(newState.puzzle)] = True;
                                                #deep += 1
                                #current = idsQueue.popleft()
                                tmp = idsQueue.get()
                                current = tmp[2]
                                #print(tmp[0])
		return current.movesSoFar

	def a_star(self, puzzle):
		
		# Queue to hold untraversed nodes
		a_starQueue = PriorityQueue()
		self.nodesVisited = 0
		self.space = 0
		# The current node/state
		current = A_starNode(puzzle, [], 0)
		
		seenPuzzleStates = {}
		seenPuzzleStates[str(current.puzzle.getGrid())] = True;
		cnt = 0
		while not(current.puzzle.won()):
			#print(current.blocking)
			self.nodesVisited += 1
			
			for m in current.getPossibleMoves():
				cnt += 1
				# Duplicate puzzle state and perform a move
				newState = copy.deepcopy(current)
				newState.puzzle.move(m.pos, m.moves)
				self.space += 1
				# If new state is unseen, add to queue and seen states list
				if ((not str(newState.puzzle) in seenPuzzleStates) or seenPuzzleStates[str(newState.puzzle)] > len(newState.movesSoFar)):
					#a_starQueue.append(A_starNode(newState.puzzle, current.movesSoFar + [m], 0))
					a_starQueue.put((current.blocking, cnt, A_starNode(newState.puzzle, current.movesSoFar + [m], current.getblocking())))
					seenPuzzleStates[str(newState.puzzle)] = True;
			#current = a_starQueue.popleft()
			tmp = a_starQueue.get()
			current = tmp[2]
		return current.movesSoFar
	def ida_star(self, puzzle):
		
		ida_starQueue = PriorityQueue()
		self.nodesVisited = 0
		self.space = 0
		# The current node/state
		current = IDA_starNode(puzzle, [], 0, 0)
		
		seenPuzzleStates = {}
		seenPuzzleStates[str(current.puzzle.getGrid())] = True;
		cnt = 0
		while not(current.puzzle.won()):
			#print(current.blocking)
			self.nodesVisited += 1
			
			for m in current.getPossibleMoves():
				cnt += 1
				# Duplicate puzzle state and perform a move
				newState = copy.deepcopy(current)
				newState.puzzle.move(m.pos, m.moves)
				self.space += 1
				# If new state is unseen, add to queue and seen states list
				if ((not str(newState.puzzle) in seenPuzzleStates) or seenPuzzleStates[str(newState.puzzle)] > len(newState.movesSoFar)):
					#a_starQueue.append(A_starNode(newState.puzzle, current.movesSoFar + [m], 0))
					ida_starQueue.put((current.thedeep, current.blocking, cnt, IDA_starNode(newState.puzzle, current.movesSoFar + [m], current.getblocking(), current.thedeep+1)))
					seenPuzzleStates[str(newState.puzzle)] = True;
			#current = a_starQueue.popleft()
			tmp = ida_starQueue.get()
			current = tmp[3]
		return current.movesSoFar
	
class BfsNode:
	def __init__(self, puzzle, movesSoFar):
		self.puzzle = puzzle
		self.movesSoFar = movesSoFar

	def getPossibleMoves(self):
		results = []
		current = self.puzzle
		for v in current.vehicles:
			for i in current.moveRange(v):
				#print('v:',v,'v')
				# Don't move if move length is 0
				if not i == 0:
						results += [Move(v.pos, i, v.orientation, v.number)]
		#print(results)						
		return results

class DfsNode:
	def __init__(self, puzzle, movesSoFar):
		self.puzzle = puzzle
		self.movesSoFar = movesSoFar

	def getPossibleMoves(self):
		results = []
		current = self.puzzle
		for v in current.vehicles:
			for i in current.moveRange(v):
				#print('v:',v,'v')
				# Don't move if move length is 0
				if not i == 0:
						results += [Move(v.pos, i, v.orientation, v.number)]
		#print(results)						
		return results

class IdsNode:
	def __init__(self, puzzle, movesSoFar, thedeep):
		self.puzzle = puzzle
		self.movesSoFar = movesSoFar
		self.thedeep = thedeep

	def getPossibleMoves(self):
		results = []
		current = self.puzzle
		for v in current.vehicles:
			for i in current.moveRange(v):
				#print('v:',v,'v')
				# Don't move if move length is 0
				if not i == 0:
						results += [Move(v.pos, i, v.orientation, v.number)]
		#print(results)						
		return results

class A_starNode:
	def __init__(self, puzzle, movesSoFar, blocking):
		self.puzzle = puzzle
		self.movesSoFar = movesSoFar
		self.blocking = blocking

	def getPossibleMoves(self):
		results = []
		current = self.puzzle
		for v in current.vehicles:
			for i in current.moveRange(v):
				#print('v:',v,'v')
				# Don't move if move length is 0
				if not i == 0:
						results += [Move(v.pos, i, v.orientation, v.number)]
		#print(results)						
		return results
	def getblocking(self):
                current = self.puzzle
                blockingcars = 0
                red_car_pos = 0
                for v in current.vehicles:      #find red car
                        if v.number == '0':
                                if v.pos[0] == 4:
                                        return 0
                                else:
                                        red_car_pos = v.pos[0]
                                break
                blockingcars = 1
                for v in current.vehicles:
                        if v.orientation == Orientations.vertical and v.pos[0] > red_car_pos:   #may block
                                if v.vType == VehicleTypes.car and (v.pos[1] == 1 or v.pos[1] == 2):    #is block
                                        blockingcars +=1
                                elif v.vType == VehicleTypes.truck and (v.pos[1] >= 0 and v.pos[1] <=2):    #is block
                                        blockingcars +=1
                return blockingcars

class IDA_starNode:
	"""Represents a single state of the BFS
	Attributes:
		puzzle (JamPuzzle):  the puzzle state this node represents
		movesSoFar (Move[]):  array of moves taken to get to the current
				state.  Holds the solution at the end, since BFs itself
				doesn't track moves so far for each state.
	getPossibleMoves(self): retrieves list of all valid moves from this
			node's state
	"""

	def __init__(self, puzzle, movesSoFar, blocking, thedeep):
		"""Constructor takes a puzzle state and list of moves taken
		so far to get there.
		"""
		self.puzzle = puzzle
		self.movesSoFar = movesSoFar
		self.blocking = blocking
		self.thedeep = thedeep

	def getPossibleMoves(self):
		"""Find the moveRange() of each vehicle in puzzle state and
		adds every move (except 0 moves) in the range for each vehicle
		to a result list of Move objects
		Return: 
			Move[]: the array of all valid moves for this node's state
		"""
		results = []
		current = self.puzzle
		for v in current.vehicles:
			for i in current.moveRange(v):
				#print('v:',v,'v')
				# Don't move if move length is 0
				if not i == 0:
						results += [Move(v.pos, i, v.orientation, v.number)]
		#print(results)						
		return results
	def getblocking(self):
                current = self.puzzle
                blockingcars = 0
                red_car_pos = 0
                for v in current.vehicles:      #find red car
                        if v.number == '0':
                                if v.pos[0] == 4:
                                        return 0
                                else:
                                        red_car_pos = v.pos[0]
                                break
                blockingcars = 1
                for v in current.vehicles:
                        if v.orientation == Orientations.vertical and v.pos[0] > red_car_pos:   #may block
                                if v.vType == VehicleTypes.car and (v.pos[1] == 1 or v.pos[1] == 2):    #is block
                                        blockingcars +=1
                                elif v.vType == VehicleTypes.truck and (v.pos[1] >= 0 and v.pos[1] <=2):    #is block
                                        blockingcars +=1
                return blockingcars
class Move:
	def __init__(self, pos, moves, orientation, number):
		self.pos = pos;
		self.moves = moves;
		self.orientation = orientation;
		self.number = number;
		
	def __str__(self):
		#print(self.orientation, self.number)
		#return "Move car at ("+str(self.pos[1])+','+str(self.pos[0])+") by "+str(self.moves)+" to ("+str(self.pos[1])+','+str(self.pos[0])+")"
                action = "< " + self.number + ", " + str(self.pos[1]) + ", " + str(self.pos[0]) + " >"
                if self.orientation == Orientations.horizontal:
                        return action + " ==> < " + self.number + ", " + str(self.pos[1]) + ", " + str(self.pos[0]+self.moves) + " >"
                else:
                        return action + " ==> < " + self.number + ", " + str(self.pos[1]+self.moves) + ", " + str(self.pos[0]) + " >"
                
