import numpy as np
import math
from .lum_dist import Angdist
from .constants import __c__, __pc_2_au__

def bscale(vsys,nx,pixel,config,hdr_info,f=4):

	vsys_tmp= vsys
	red_	= vsys_tmp/__c__
	dist	= Angdist()
	hdr_ori	= hdr_info.cube_hdr()
	others	= config['others']
	frame	= others.getint('frame',None)
	dist_pc	= others.getfloat('distance',None)

	vcor = 0
	if frame is not None and frame in [1,2,3]:
		if 'SPECSYS' in hdr_ori and hdr_ori['SPECSYS'] == 'LSRK':
			# This means we need to first transform from LSRK to Heliocentric
			vlsrk2helio = -1 * dist.vcor(corr_vel=True,header=hdr_ori,frame='Helio2LSRK')
			vsys_tmp = vsys_tmp + vlsrk2helio
			print('VLSRK -> VHELIO = +%s km/s'%vlsrk2helio)

		# change the reference frame from Heliocentric to another frame.
		# vcor is 0 km/s if corr_vel is False.
		vcor=dist.vcor(corr_vel=True,header=hdr_ori,frame='Helio2CMB')
		print('VHELIO -> VCMB = %s km/s'%vcor)

	red_=(vsys_tmp+vcor)/__c__
	dL,scale_pc_arc=dist.comv_distance(red_)


	if dist_pc is not None:
		scale_pc_arc=dist_pc/206265.
	bar_scale_arc0 = (nx//f)*pixel

	if scale_pc_arc < 1:
		if dist_pc is not None:
			bar_scale_au = int(bar_scale_arc0)
			bar_scale_u=int(bar_scale_au*scale_pc_arc*__pc_2_au__)
			return bar_scale_au,bar_scale_u,'AU'
		else:
			bar_scale_au = int(bar_scale_arc0)
			bar_scale_u=int(bar_scale_au*scale_pc_arc)
			return bar_scale_au,0,'pc'

	if bar_scale_arc0 < 1:
		round_0=abs(math.floor(np.log10(bar_scale_arc0)))
		bar_scale_arc0=round(bar_scale_arc0,round_0)
		bar_scale_arc=bar_scale_arc0

	elif bar_scale_arc0 // 10 == 0:
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
