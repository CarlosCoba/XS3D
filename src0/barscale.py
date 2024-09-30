import numpy as np
from src0.lum_dist import Angdist
from src0.constants import __c__

def bscale(vsys,nx,pixel,f=4):
	red_ = vsys/__c__
	dist=Angdist(red_)
	dL,scale_pc_arc=dist.comv_distance()
	bar_scale_arc0 = (nx//f)*pixel
	
	if scale_pc_arc < 1:
		bar_scale_au = int(bar_scale_arc0)
		bar_scale_u=int(bar_scale_au*scale_pc_arc*206265)
		return bar_scale_au,0,'pc'
		
	if bar_scale_arc0 // 10 == 0:
		round_int=int(abs(np.log10(bar_scale_arc0)))
		bar_scale_arc=int(round(bar_scale_arc0,round_int))
	else:	
		bar_scale_arc = 10*( bar_scale_arc0// 10 )
		bar_scale_arc = int(round(bar_scale_arc,0))
	bar_scale_pc = bar_scale_arc*scale_pc_arc
	
	# kpc units
	if bar_scale_pc// 1000>0 or bar_scale_pc//500==1:
		unit = 'kpc'
		bar_scale = bar_scale_pc / 1000
		bar_scale_u = bar_scale
		bar_scale_u=round(bar_scale_u,1)			
	# pc units		
	#if bar_scale_pc // 1000 == 0:
	else:
		unit = 'pc'	
		bar_scale =  bar_scale_pc
		bar_scale_u = bar_scale
		bar_scale_u=int(round(bar_scale_u))			


	return bar_scale_arc,bar_scale_u,unit
