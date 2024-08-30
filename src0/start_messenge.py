import numpy as np
np.set_printoptions(precision=4,suppress=True)
from src0.read_config import config_file

class Print:
	def __init__(self):
		self.n=36 
		self.deli="-"*self.n
			
	def guess_vals(self,galaxy,guess,vmode):
		PA,INC,X0,Y0,VSYS,PHI_B = 	guess[0],guess[1],guess[2],guess[3], guess[4], guess[5] 

		print(self.deli)		
		print("Guess values for %s"%galaxy)
		print("PA:\t\t %s"%PA)
		print("INC:\t\t %s"%INC)
		print("X0,Y0:\t\t %s,%s"%(round(X0,2),round(Y0,2)))
		print("VSYS:\t\t %s"%round(VSYS,2))
		if vmode == "bisymmetric" :
			print("PHI_BAR:\t %s"%PHI_B)			
		print("MODEL:\t\t %s"%vmode)
		print(self.deli)		

	def __call__(self):
		print(self.deli)		
		print("---------------XS3D-----------------")
		

	def out(self,hdr,value,tab="\t"):
		print(self.deli)		
		print(f'{hdr}:{tab}{value}')
		
	def cubehdr(self,hdr):
		print(self.deli)		
		print(f"Cube dims:\t{hdr.nz}x{hdr.ny}x{hdr.ny} pix")
		print(f"Rest frame eline: \t {hdr.eline}")	
		print(f"Channel width: \t {hdr.cdelt3_kms} km/s")		

	def status(self,msn,line=False):
		print(self.deli)			
		m=len(msn)
		space=""
		if m < self.n:
			space="."*(self.n-m-1)
		print(f'{msn} {space}')	
		if line: print(self.deli)			


