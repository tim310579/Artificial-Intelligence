from enum import Enum, IntEnum

class VehicleTypes(IntEnum):
	car = 2
	truck = 3
class Orientations(IntEnum):
	horizontal = 0
	vertical = 1
#class Number(IntEnum):


class JamPuzzle:
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
		orient = veh.orientation
		newPosList = list(veh.pos)
		newPosList[orient] += moves
		veh.pos = tuple(newPosList)


	def moveRange(self, veh):
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
		for v in self.vehicles:
			if v.pos == pos:
				return v
		return None

	def won(self):
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
	
	def __init__(self, pos, orientation, vType, number):
		self.pos = pos
		self.orientation = orientation
		self.vType = vType
		self.number = number

	def coveredUnits(self):
		
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
