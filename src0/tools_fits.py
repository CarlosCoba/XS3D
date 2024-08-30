from astropy.io import fits

def array_2_fits(data,out,galaxy,vmode, kwargs):
	if type(kwargs) not in [dict,list]:
		print("XookSuut: bad header assignation");quit()
	
	hdu = fits.PrimaryHDU(data)
	for k in range(len(kwargs)):
		hdr_k = kwargs["%s"%k]
		hdr, val = hdr_k
		hdu.header['%s'%hdr] = val
	
	hdu.writeto("%smodels/%s.%s.chain.fits.gz"%(out,galaxy,vmode),overwrite=True)



