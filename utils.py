
class AvgMeter() :
	def __init__(self) :
		self.reset()

	def reset(self) :
		self.sum = 0
		self.count = 0

	def __call__(self, val = None, reset = False) :
		if val is not None :
			self.sum += val
			self.count += 1
		result = 0
		if self.count > 0 :
			result = self.sum / self.count
		if reset :
			self.reset()
		return result
