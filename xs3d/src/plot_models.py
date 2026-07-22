import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import matplotlib.ticker as ticker
from matplotlib.offsetbox import AnchoredText


from .axes_params import axes_ambient as axs
from .cbar import colorbar as cb
from .colormaps_CLC import vel_map
from .constants import __c__

#params =   {'text.usetex' : True }
#plt.rcParams.update(params)

cmap = vel_map()

def plot_kin_models(galaxy,vmode,const,best,result,out):

	R=abs(best['radius'])
	nrings=len(R)	
	[v_sys,inc,pa,x_center,y_center,phi_bar,rmax]=const['v_sys'],const['inc'],const['pa'],const['x_center'],const['y_center'],const['phi_bar'],const['rmax']

	scalar_fields = ["v_rot", "v_rad", "v_2t", "v_2r", "v_disp"]
	vels = {k:best[k] for k in scalar_fields}
	vrot = vels['v_rot']
	vrad = vels['v_rad'] if vmode != 'bisymmetric' else 	vels['v_2r']
	vtan = vels['v_2t']
	vdisp = vels['v_disp']
	v_nc_max = np.max( abs(np.concatenate([vrad.ravel(), vtan.ravel()])))
	
	p = ['v_rot_r','v_rad_r','v_2t_r','v_disp_r']
	evrot,evrad,evtan,evdisp=[],[],[],[]
	ev = [evrot,evrad,evtan,evdisp]
	for k,v in enumerate(p):
		for n in range(len(vrot)):
			try:
				error=result.params[v+f'{n}'].stderr
				(ev[k]).append(error if error is not None else 0)
			except(KeyError):
				(ev[k]).append(0)		
	
	width, height = 10, 6 # width [cm]
	cm_to_inch = 0.393701 # [inch/cm]
	figWidth = width * cm_to_inch # width [inch]
	figHeight = height * cm_to_inch # width [inch]

	if vmode=='circular':
		fig = plt.figure(figsize=(figWidth, figHeight), dpi = 300)
		gs2 = fig.add_gridspec(nrows=1, ncols=1, left=0.10, right=0.85, wspace=0, bottom=0.17, top = 0.95)
		ax0=plt.subplot(gs2[0,0])
	else:
		width, height = 10, 9 # width [cm]
		figWidth = width * cm_to_inch # width [inch]
		figHeight = height * cm_to_inch # width [inch]
		heights = [1, 0.6]
	
		fig = plt.figure(figsize=(figWidth, figHeight), dpi = 300)		
		gs = gridspec.GridSpec(2, 1, height_ratios=heights)
		gs.update(left=0.13, right=0.85,top=0.97,bottom=0.10, hspace = 0.15, wspace = 0)
		ax0 = plt.subplot(gs[0,0])
		ax1 = plt.subplot(gs[1,0])
		
	
	# is it the axes in arcsec or arcmin ?
	rnorm=1
	if np.max(R)>80:
		rnorm=60
		rlabel='arcmin'
	else:
		rlabel='arcsec'

	R=R/rnorm
	delta_r = R[-1]-R[-2]
	max_r = np.max(R)
		
	axs(ax0, rotation='horizontal',remove_axis_lines = True, fontsize_ticklabels=10)


	ax0.errorbar(R, vdisp, yerr=evdisp, color = "#db6d52", label = "$\sigma_\mathrm{gas}$",  fmt='o', mfc = '#db6d52', mec = '#170a06',ms = 4,mew = 0.5, ecolor='#db6d52', lw=1, ls = ':', capsize=2)
	ax0.errorbar(R, vrot, yerr=evrot, color = "#362a1b", label = "$\mathrm{V_{t}}$",  fmt='o', mfc = '#362a1b', mec = '#170a06',ms = 4,mew = 0.5, ecolor='#362a1b', lw=1, ls = ':', capsize=2)	
	if vmode != 'circular':
		ax1.errorbar(R, -1e4*vdisp, yerr=evdisp, color = "#db6d52", label = "$\sigma_\mathrm{gas}$",  fmt='o', mfc = '#db6d52', mec = '#170a06',ms = 4,mew = 0.5, ecolor='#db6d52', lw=1, ls = ':', capsize=2)
		ax1.errorbar(R, -1e4*vrot, yerr=evrot, color = "#362a1b", label = "$\mathrm{V_{t}}$",  fmt='o', mfc = '#362a1b', mec = '#170a06',ms = 4,mew = 0.5, ecolor='#362a1b', lw=1, ls = ':', capsize=2)
	
	if vmode == "radial":
		ax1.errorbar(R, vrad, yerr=evrad, color = "#c73412", label = "$\mathrm{V_{r}}$", fmt='o', mfc = '#c73412', mec = '#170a06',ms = 4,mew = 0.5, ecolor='#c73412', lw=1, ls = ':', capsize=2)

	if vmode == "vertical":
		ax1.errorbar(R, vrad, yerr=evrad, color = "#b47b50", label = "$\mathrm{V_{z}}$", fmt='o', mfc = '#b47b50', mec = '#170a06',ms = 4,mew = 0.5, ecolor='#b47b50', lw=1, ls = ':', capsize=2)

	if vmode == "bisymmetric":
		ax1.errorbar(R, vrad, yerr=evrad, color = "#c73412", label = "$\mathrm{V_{2,r}}$", fmt='o', mfc = '#c73412', mec = '#170a06',ms = 4,mew = 0.5, ecolor='#c73412', lw=1, ls = ':', capsize=2)
		ax1.errorbar(R, vtan, yerr=evtan, color = "#2fa7ce", label = "$\mathrm{V_{2,t}}$", fmt='o', mfc = '#2fa7ce', mec = '#170a06',ms = 4,mew = 0.5, ecolor='#2fa7ce', lw=1, ls = ':', capsize=2)

	# Move the left and bottom spines to x = 0 and y = 0, respectively.
	ax0.spines[["left", "bottom"]].set_position(("data", 0))
	# Hide the top and right spines.
	ax0.spines[["top", "right"]].set_visible(False)

	ax0.plot(1, 0, ">k", transform=ax0.get_yaxis_transform(), clip_on=False)
	ax0.plot(0, 1, "^k", transform=ax0.get_xaxis_transform(), clip_on=False)

	ax0.set_ylim(0, 30*(np.max(vrot)//30)+40)
	ax0.xaxis.set_major_locator(ticker.MaxNLocator(5))
	ax0.yaxis.set_major_locator(ticker.MaxNLocator(5))
	
	dashline = (5, (10, 3))
	ax0.plot([0,max_r],[0,0],color = "k",linestyle=dashline, alpha = 0.6,linewidth = 0.5)
	ax0.set_ylabel('$\mathrm{Velocity\,(km~s^{-1})}$',fontsize=10)

	if vmode != 'circular':
		ax1.legend(loc = "upper left", fontsize=10, bbox_to_anchor=(1,1), ncol=1, bbox_transform=ax0.transAxes, labelspacing=0.7, handlelength=1, handletextpad=0.3,columnspacing=0.8,borderaxespad=0.1,frameon=False)	
		ax0.set_xlim(-delta_r*0.1, max_r+delta_r*0.1)
		ax1.set_xlim(-delta_r*0.1, max_r+delta_r*0.1)	
		ax1.plot([0,max_r],[0,0],color = "k",linestyle=dashline, alpha = 1,linewidth = 0.5)				
		axs(ax1, remove_xticks= False, rotation = 'horizontal', fontsize_ticklabels=10)
		ax1.grid(visible = True, which = "major", axis = "both", color='gray', linestyle='-', linewidth=0.5, zorder = 1, alpha = 0.5)
		ax1.xaxis.set_major_locator(ticker.MaxNLocator(5))		
		ax1.yaxis.set_major_locator(ticker.MaxNLocator(3))
		ax1.set_ylim(-v_nc_max -4 , v_nc_max + 4)
		ax1.set_xlabel(f'r ({rlabel})',fontsize=10)
		ax1.set_ylabel("$\mathrm{V_{NC}~(km\,s^{-1})}$",fontsize = 10)					
	else:
		ax0.legend(loc="upper left", fontsize = 10, bbox_to_anchor=(1,1), ncol=1, labelspacing=0.7, handlelength=1, handletextpad=0.3,columnspacing=0.8,borderaxespad=0.1,frameon=False)	
		ax0.set_xlabel(f'r ({rlabel})',fontsize=10)	
				
	fig.tight_layout()
	plt.savefig("%sfigures/kin_%s_disp_%s.png"%(out,vmode,galaxy))
	#plt.clf()
	plt.close()
