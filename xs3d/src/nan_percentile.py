import numpy as np
'''
This function significantly improves the computation time of nanpercentile !
source: https://krstn.eu/np.nanpercentile()-there-has-to-be-a-faster-way/
'''


def nan_percentile(arr, q):
	'''
	arr: 3D or 1D array.
		 Percentile is always computed along the 0 axis.
	q: list of percentiles i.e, [10,50,90]
	'''
	type_q=type(q)
	if type_q != list:
		q = [q]

	# If array is 1D then reshape to 3D
	oned=False
	if np.ndim(arr)==1:
		oned=True
		arr=arr.reshape((len(arr),1,1))
	
	# valid (non NaN) observations along the first axis
	valid_obs = np.sum(np.isfinite(arr), axis=0)
	# replace NaN with maximum
	max_val = np.nanmax(arr)
	arr[np.isnan(arr)] = max_val
	# sort - former NaNs will move to the end
	arr = np.sort(arr, axis=0)

	# loop over requested quantiles
	if type(q) is list:
		qs = []
		qs.extend(q)
	else:
		qs = [q]
	if len(qs) < 2:
		quant_arr = np.zeros(shape=(arr.shape[1], arr.shape[2]))
	else:
		quant_arr = np.zeros(shape=(len(qs), arr.shape[1], arr.shape[2]))

	result = []
	for i in range(len(qs)):
		quant = qs[i]
		# desired position as well as floor and ceiling of it
		k_arr = (valid_obs - 1) * (quant / 100.0)
		f_arr = np.floor(k_arr).astype(np.int32)
		c_arr = np.ceil(k_arr).astype(np.int32)
		fc_equal_k_mask = f_arr == c_arr

		# linear interpolation (like numpy percentile) takes the fractional part of desired position
		floor_val = _zvalue_from_index(arr=arr, ind=f_arr) * (c_arr - k_arr)
		ceil_val = _zvalue_from_index(arr=arr, ind=c_arr) * (k_arr - f_arr)

		quant_arr = floor_val + ceil_val
		quant_arr[fc_equal_k_mask] = _zvalue_from_index(arr=arr, ind=k_arr.astype(np.int32))[fc_equal_k_mask]  # if floor == ceiling take floor value

		if oned:
			quant_arr=np.squeeze(quant_arr)
		result.append(quant_arr)

	if len(q)>1:
		return result
	else:
		return quant_arr
	
def _zvalue_from_index(arr, ind):
	"""private helper function to work around the limitation of np.choose() by employing np.take()
	arr has to be a 3D array
	ind has to be a 2D array containing values for z-indicies to take from arr
	See: http://stackoverflow.com/a/32091712/4169585
	This is faster and more memory efficient than using the ogrid based solution with fancy indexing.
	"""
	# get number of columns and rows
	_,nC,nR = arr.shape

	# get linear indices and extract elements with np.take()
	#idx = nC*nR*ind + nR*np.arange(nR)[:,None] + np.arange(nC)
	idx = nC*nR*ind + np.arange(nC*nR).reshape((nC,nR))	
	return np.take(arr, idx)
	
	
if __name__ == '__main__':
	#
	# CLC: This test the speed of nanpercentile vs numpy
	# create array of shape(5,100,100) - image of size 10x10 with 5 layers
	ny,nx,nz=447,448,100
	ny,nx,nz=1,1,10	
	Ntot=int(ny*nx*nz)
	test_arr = np.random.randint(0, 10000, Ntot).reshape(nz,ny,nx).astype(np.float32)
	np.random.shuffle(test_arr)
	# place random NaN
	rand_NaN = np.random.randint(0, Ntot, 500).astype(np.float32)
	for r in rand_NaN:
		test_arr[test_arr == r] = np.NaN
	
	test_arr*=np.pi	
	input_arr=test_arr
	result=np.nanpercentile(input_arr, q=[11], axis=0);print(result, result.shape)
	result=nan_percentile(test_arr, q=[11]);print(result, result.shape)

