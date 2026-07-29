import numpy as np
import math
from .lum_dist import Angdist
from .constants import __c__, __pc_2_au__
from .start_messenge import Print

def bscale(const,config,hdr_info,f=4):

	P = Print()
	vsys_tmp= const['v_sys']
	red_	= vsys_tmp/__c__
	dist	= Angdist()
	hdr_cube	= hdr_info.cube_hdr()
	others		= config['others']
	frame_id	= others.getint('frame',0)
	dist_pc		= others.getfloat('distance',None)

	pixel	= hdr_info.scale
	nx		= hdr_info.nx

	# bar scale
	highz	= config['high_z']
	redshift= highz.getfloat('redshift',0)
	if redshift != 0:
		vsys_tmp = vsys_tmp + redshift*__c__
						
	vcor = 0					
	frame_name = {0: 'Hubble Flow', 1: 'CMB', 2: 'Galactocentric', 3: 'Local Group'}
	frame_dic = {1:'Helio2CMB', 2: 'Helio2Gal', 3: 'Helio2LG'}
	if frame_id is not None and frame_id in [1,2,3]:
		if 'SPECSYS' in hdr_cube and hdr_cube['SPECSYS'] == 'LSRK':
			# This means we need to first transform from LSRK to Heliocentric
			vlsrk2helio = -1 * dist.vcor(corr_vel=True,header=hdr_cube,frame='Helio2LSRK')
			vsys_tmp = vsys_tmp + vlsrk2helio
			print('  VLSRK  -> VHELIO = +%s km/s'% round(vlsrk2helio,3))
			
		frm	= frame_dic[frame_id] 
		# change the reference frame from Heliocentric to another frame.
		# vcor is 0 km/s if corr_vel is False.
		vcor = dist.vcor(corr_vel=True,header=hdr_cube,frame=frm)
		print('  VHELIO -> %s = %s km/s'%(frame_dic[frame_id][6:], round(vcor,3)))
	
	red_ = (vsys_tmp + vcor) / __c__
	dL,scale_pc_arc = dist.comv_distance(red_)


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

	outs = {'frame': frame_id, 'bar_scale_arc': bar_scale_arc, 'bar_scale_u': bar_scale_u, 'unit': unit, 'dL': dL, 'scale_pc_arc': scale_pc_arc }
	
	#if dist_pc is None:
	if frame_id !=0 and vcor != 0:	
		P.out('frame', frame_name[frame_id])
		P.out('scale', '%s pc/arcs'%round(scale_pc_arc,3))
		P.out('Lum. Dist.', '%s Mpc'%round(dL,3))
	return outs
