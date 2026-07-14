def get_pay(hours, rate):
	"""Return gross pay given hours and hourly rate.

	This simple implementation multiplies hours by rate. If hours or rate
	are not numbers a TypeError will be raised.
	"""
	return hours * rate