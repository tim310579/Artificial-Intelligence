"""This module includes the JamPuzzle class, which represents an instance of the Traffic
Jam Puzzle.  It supports all moves, legal or not, and will track the state of the puzzle
for any moves made.
It also includes the Vehicle class, which represents a single vehicle with a position,
orientation, and vehicle type.
(c) 2016 Gary Chen
"""
from enum import Enum, IntEnum


"""Some useful enums!
	
VehicleTypes holds vehicle length for each type.  Orientations just 
tracks orientation of a vehicle, useful for dereferencing position tuples.
"""
class VehicleTypes(IntEnum):
	car = 2
	truck = 3
class Orientations(IntEnum):
	horizontal = 0
	vertical = 1
#class Number(IntEnum):


class JamPuzzle:
	"""Represents an instance of the game
	Attributes:
		gridSizeX (int):  width of the grid
		gridSizeY (int):  height of the grid
		doorPos (int): 	position of the door (0 indexed)
		vehicles (Vehicle[]]): 	a list of Vehicle objects representing
			the initial state of the game
	"""
	def __init__(self, gridSizeX, gridSizeY, doorPos, vehicles):
		self.gridSizeY = gridSizeY
		self.gridSizeX = gridSizeX
		self.doorPos = doorPos
		self.vehicles = vehicles

	def getSizeTuple(self):
		"""Returns grid sizes as an (x, y) tuple
		Return: 
			(int, int):  tuple representing (width, height) of puzzle grid
		"""
		return (self.gridSizeX, self.gridSizeY)
	
	def getGrid(self):
		"""Returns a 2D char list representation of the state of the puzzle, not
			including the exit door's position.  Each car is represented by a
			letter in all the sapces it occupies; blank spaces are '_'.  The goal 
			car is the last car found that is aligned with the exit door, and is 
			represented by '#'.
			Return:
				grid = String[][]:  2D string array graphically depicting puzzle state
		"""  
		#symbol = ord('A')
		symbol = ord('1')
		grid = [["_" for y in range(self.gridSizeY)] for x in range(self.gridSizeX)]
		for v in self.vehicles:
			# iterate through each vehicle, assigning it a symbol and replacing its
			# covered locations with that symbol in the grid
			tempSymbol = chr(symbol)
			
			if v.pos[1] == self.doorPos and v.orientation == Orientations.horizontal:
				#print(v.pos[1])
				tempSymbol = '0'
			else:
				symbol += 1
			if symbol == 58: symbol += 7
			locs = v.coveredUnits()
			#print(locs)
			for l in locs:
				#print(l,"lll")
				grid[l[0]][l[1]] = tempSymbol
		return grid


	def move(self, pos, moves):
		"""Wrapper for moveVehicle()
		Args:
			pos ((int, int)):  position of vehicle to move (x, y)
			moves (int):  number of moves to move vehicle
		"""
		v = self.getVehicleAt(pos)
		if v == None:
			raise Exception("Can't move vehicle; not found", pos)
		self.moveVehicle(v, moves)

	
	def moveVehicle(self, veh, moves):
		"""Moves the vehicle by an amount in the directions of its 
			orientation .  Does not check for obstacles.
			
		Args:
			veh (Vehicle):	the vehicle to be moved
			moves (int):	amount to move (can be negative)
		"""
		orient = veh.orientation
		newPosList = list(veh.pos)
		newPosList[orient] += moves
		veh.pos = tuple(newPosList)


	def moveRange(self, veh):
		"""Finds legal move range of a vehicle in current puzzle.
		Args:
			veh (Vehicle):  the vehicle to check
		Return:
			(range): a range from min moves to max moves + 1
		"""

		minMove = 0
		# iterate over spaces behind to check
		for i in range(-1, -veh.pos[veh.orientation]-1, -1):

			# Only way to change a value in a tuple by index :/
			newPosList = list(veh.pos)
			newPosList[veh.orientation] += i
			newPosTuple = tuple(newPosList)

			blocked = False
			for v in self.vehicles:
				if newPosTuple in v.coveredUnits():
					blocked = True
					break
			if blocked:
				break
			else:
				minMove = i

		maxMove = 0
		# iterate over spaces ahead to check, not pos of vehicle.  Accounts for length of vehicle
		for j in range(veh.vType, self.getSizeTuple()[veh.orientation]-veh.pos[veh.orientation]):
			# j is # of spaces ahead of vehicle position to check! 
			# not position to check, or # of moves
			newPosList = list(veh.pos)
			newPosList[veh.orientation]+=j
			newPosTuple = tuple(newPosList)

			blocked = False
			for v in self.vehicles:
				if newPosTuple in v.coveredUnits():
					blocked = True
					break
			if blocked:
				break
			else:
				maxMove = j - veh.vType + 1

		return range(minMove, maxMove+1)


	def getVehicleAt(self, pos):
		"""Retrieves vehicle at given position in puzzle
		Args:
			pos ((int, int)):  position of upper-left part of vehicle to find
		Return:
			v (Vehicle):  the found vehicle, if it exists
			None:  if no vehicle was found at the position
		"""
		for v in self.vehicles:
			if v.pos == pos:
				return v
		return None

	def won(self):
		"""Checks if game is in a win-state, i.e. a vertical vehicle is adjacent
		to the door on the top edge of the puzzle grid
		Return:
			(bool):  boolean representing if game is in a win state or not
		"""
		v = self.getVehicleAt((4, self.doorPos))
		#print('haha:', v,'over')
		if v != None and v.orientation == Orientations.horizontal:
			return True
		return False

	def __str__(self):
		result = "  " * self.doorPos + " " + "  " * (self.gridSizeX - self.doorPos - 1) + "\n"
		grid = self.getGrid()
		result += "\n".join([" ".join([grid[x][y] for x in range(self.gridSizeX)]) for y in range(self.gridSizeY)]) + "\n"
		return result

	def __eq__(self, b):
		return self.getGrid() == b.getGrid()


class Vehicle:
	"""Represents a single vehicle instance
	Attributes:
		pos ((int, int)):  the (x, y) position of the top-left square of the vehicle
		orientation (Orientations):  an Orientations enum value representing direction
			the vehicle is facing (vertical or horizontal; backwards or forwards doesn't matter)
		vType (VehicleTypes): a VehicleTypes enum value representing type of vehicle, either
			car or truck.  Also represents length of vehicle.
	"""
	def __init__(self, pos, orientation, vType, number):
		self.pos = pos
		self.orientation = orientation
		self.vType = vType
		self.number = number

	def coveredUnits(self):
		"""Returns list of all locations covered by this vehicle.
		Return:
			result ((int, int)[]):  Array of tuples representing all coordinates this
				vehicle must cover, given its position and length.
		"""
		if self.orientation == Orientations.vertical:
			result = [(self.pos[0], self.pos[1] + i) for i in range(int(self.vType))]
		if self.orientation == Orientations.horizontal:
			result = [(self.pos[0] + i, self.pos[1]) for i in range(int(self.vType))]
		return result

	def __str__(self):
		orientTxt = "Horizontal" if self.orientation == Orientations.horizontal else "Vertical"
		vehTxt = "Car" if self.vType == VehicleTypes.car else "Truck"
		positions = str(self.coveredUnits())
		return orientTxt + " " + vehTxt + " at (" + str(self.pos[0]) + "," + str(self.pos[1]) + ") covering " + positions
