import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

from matplotlib.offsetbox import AnchoredText
from src0.axes_params import axes_ambient as axs 
from src0.cbar import colorbar as cb
from src0.colormaps_CLC import vel_map


def vmin_vmax(data,pmin=5,pmax=98,base=None):
	vmin,vmax=np.nanpercentile(np.unique(data),pmin),np.nanpercentile(np.unique(data),pmax)
	if base is not None:
		vmin,vmax=(vmin//base+1)*base,(vmax//base+1)*base
	return vmin,vmax

def zero2nan(data):
	data[data==0]=np.nan
	return data
	
prng =  np.random.RandomState(123)

#params =   {'text.usetex' : True }
#plt.rcParams.update(params) 
cmap = vel_map()

# These are ad-hoc colors. I like this.
list_fancy_colors = ["#362a1b", "#fc2066", "#f38c42", "#4ca1ad", "#e7af35", "#85294b", "#915f4d", "#86b156", "#b74645", "#2768d9", "#cc476f", "#889396", "#6b5b5d", "#963207"]

def plot_kin_models_h(galaxy,vmode,momms_mdls,R,Sigma,eSigma,Ck,Sk,e_Ck,e_Sk,VSYS,ext, m_hrm, out):
	mom0_mdl,mom1_mdl,mom2_mdl_kms,mom2_mdl_A,cube_mdl,velmap_intr,sigmap_intr= momms_mdls
	c1 = Ck[0]
	e_c1 = e_Ck[0]
	s_1 = Sk[0]
	Rnoncirc = R*(s_1/s_1) 



	width, height = 14.0, 5 # width [cm]
	#width, height = 3, 15 # width [cm]
	cm_to_inch = 0.393701 # [inch/cm]
	figWidth = width * cm_to_inch # width [inch]
	figHeight = height * cm_to_inch # width [inch]
  
	fig = plt.figure(figsize=(figWidth, figHeight), dpi = 300)
	#nrows x ncols
	widths = [1.2, 1.2, 0.4, 2]
	heights = [1, 0.6]
	gs = gridspec.GridSpec(2, 4,  width_ratios=widths, height_ratios=heights)
	gs.update(left=0.06, right=0.99,top=0.81,bottom=0.14, hspace = 0.03, wspace = 0)

	ax0 = plt.subplot(gs[0:2,0])
	ax1 = plt.subplot(gs[0:2,1])

	ax3 = plt.subplot(gs[0,3])
	ax4 = plt.subplot(gs[1,3])



	# intrinsic velocity
	velmap_intr=zero2nan(velmap_intr)
	velmap=velmap_intr-VSYS
	
	vmin = abs(np.nanmin(velmap))
	vmax = abs(np.nanmax(velmap))
	max_vel = np.nanmax([vmin,vmax])	
	vmin = -(max_vel//50 + 1)*50
	vmax = (max_vel//50 + 1)*50

	im0=ax0.imshow(velmap,origin='lower',cmap=cmap,extent=ext,vmin=vmin,vmax=vmax,aspect='auto')
	
	
	# intrinsic velocity dispersion
	sigmap_intr=zero2nan(sigmap_intr)
	vmin = abs(np.nanmin(sigmap_intr))
	vmax = abs(np.nanmax(sigmap_intr))
	max_vel = np.nanmax([vmin,vmax])	
	vmin = -(max_vel//25 + 1)*25
	vmax = (max_vel//25 + 1)*25
	vmin,vmax=vmin_vmax(sigmap_intr,base=10)
	im1=ax1.imshow(sigmap_intr,origin='lower',cmap=cmap,extent=ext,vmin=vmin,vmax=vmax,aspect='auto')		

	
	# If you dont want to add these lines ontop of the 2D plots, then comment the following.
	# it seems that  calling astropy.convolution causes that some package is downloaded each time this is run, why ?
	"""
	from astropy.convolution import Gaussian2DKernel
	from astropy.convolution import convolve
	import matplotlib
	matplotlib.rcParams['contour.negative_linestyle']= 'solid'

	kernel = Gaussian2DKernel(x_stddev=4)
	vloss = convolve(vel_ha, kernel)
	z = np.ma.masked_array(vloss.astype(int),mask= np.isnan(vel_ha) )
	N = vmax // 50
	ax0.contour(z, levels = [i*50 for i in np.arange(-N,N+1)], colors = "k", alpha = 1, zorder = 1e3, extent = ext, linewidths = 0.6)
	ax1.contour(z, levels = [i*50 for i in np.arange(-N,N+1)], colors = "k", alpha = 1, zorder = 1e3, extent = ext, linewidths = 0.6)
	"""

	axs(ax0)
	axs(ax1,remove_yticks= True)


	ax0.set_ylabel('$\mathrm{ \Delta Dec~(arcsec)}$',fontsize=10,labelpad=1)
	ax0.set_xlabel('$\mathrm{ \Delta RA~(arcsec)}$',fontsize=10,labelpad=1)
	ax1.set_xlabel('$\mathrm{ \Delta RA~(arcsec)}$',fontsize=10,labelpad=1)


	txt = AnchoredText('$\mathrm{V}_{intrinsic}$', loc="upper left", pad=0.1, borderpad=0, prop={"fontsize":11},zorder=1e4);ax0.add_artist(txt)
	txt = AnchoredText('$\sigma_{intrinsic}$', loc="upper left", pad=0.1, borderpad=0, prop={"fontsize":11},zorder=1e4);ax1.add_artist(txt)

	ax3.plot(R,c1, color = "#362a1b",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{c_{1}}$")
	ax3.fill_between(R, c1-e_c1, c1+e_c1, color = "#362a1b", alpha = 0.3, linewidth = 0)
	
	ax3.plot(R,Sigma, color = "gold",linestyle='--', alpha = 1, linewidth=0.8, label = "$\sigma_{intr}$")
	ax3.fill_between(R, Sigma-eSigma, Sigma+eSigma, color = "gold", alpha = 0.3, linewidth = 0)

	ax3.set_ylim(0, max(c1) + 50)
	ax3.set_xlim(0, max(R))
	axs(ax3, remove_xticks= True)


	# plot the 1D velocities with the predefined colors
	ax4.plot(R,-1e4*Sigma, color = "gold",linestyle='--', alpha = 1, linewidth=0.8, label = "$\sigma_{intr}$")
	ax4.plot(R,-1e4*np.ones(len(c1)), color = "#362a1b",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{c_{1}}$")
	ax4.plot([0,np.nanmax(R)],[0,0],color = "k",linestyle='-', alpha = 0.6,linewidth = 0.3)
	if 2*m_hrm < len(list_fancy_colors):
		for i in range(m_hrm):
			color = list_fancy_colors
			if i >= 1: 
				ax4.plot(Rnoncirc,Ck[i], color = color[i],linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{c_{%s}}$"%(i+1))
				ax4.fill_between(Rnoncirc, Ck[i]-e_Ck[i], Ck[i]+e_Ck[i], color = color[i], alpha = 0.3, linewidth = 0)

			ax4.plot(Rnoncirc,Sk[i], color = color[i+m_hrm],linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{s_{%s}}$"%(i+1))
			ax4.fill_between(Rnoncirc, Sk[i]-e_Sk[i], Sk[i]+e_Sk[i], color = color[i+m_hrm], alpha = 0.3, linewidth = 0)

	else:
		ax4.clear()
		#ax4.plot(R,c1*0, color = "#362a1b",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{c_{1}}$")
		#ax4.plot([0,np.nanmax(R)],[0,0],color = "k",linestyle='-', alpha = 0.6,linewidth = 0.3)
		# pick up a random color
		import random
		colors = []
		for name, hex in matplotlib.colors.cnames.items():
			colors.append(name)

		n = len(colors)
		for i in range(m_hrm):
			k1 = prng.randint(0, n-1)
			k2 = prng.randint(0, n-1)
			if i >= 1: 
				ax4.plot(Rnoncirc,Ck[i], color = colors[k1],linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{c_{%s}}$"%(i+1))
				ax4.fill_between(Rnoncirc, Ck[i]-e_Ck[i], Ck[i]+e_Ck[i], color = colors[k1], alpha = 0.3, linewidth = 0)

			ax4.plot(Rnoncirc,Sk[i], color = colors[k2],linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{s_{%s}}$"%(i+1))
			ax4.fill_between(Rnoncirc, Sk[i]-e_Sk[i], Sk[i]+e_Sk[i], color = colors[k2], alpha = 0.3, linewidth = 0)



	axs(ax4, fontsize_ticklabels = 6)
	vmin_s1 = 5*abs(np.nanmin(Sk[0]))//5
	vmax_s1 = 5*abs(np.nanmax(Sk[0]))//5
	max_vel_s1 = np.nanmax([vmin_s1,vmax_s1])

	ax4.set_ylim(-max_vel_s1 -4 , max_vel_s1 + 4)
	ax4.set_xlim(0, max(R))
	ax4.set_xlabel("$\mathrm{r~(arcsec)}$", labelpad = 2, fontsize = 10)
	ax3.set_ylabel("$\mathrm{c_{1} (km/s)}$",fontsize = 10,labelpad=2)
	ax4.set_ylabel("$\mathrm{v_{non-circ}}$ \n $\mathrm{(km/s)}$",fontsize = 10,labelpad=2)


	cb(im0,ax0,orientation = "horizontal", colormap = cmap, bbox=(0.10,1.1,0.8,1),width = "100%", height = "5%",label_pad = -26, label = "$\mathrm{(km~s^{-1})}$",labelsize=10, ticksfontsize = 8)
	cb(im1,ax1,orientation = "horizontal", colormap = cmap, bbox=(0.10,1.1,0.8,1),width = "100%", height = "5%",label_pad = -26, label = "$\mathrm{(km~s^{-1})}$",labelsize=10, ticksfontsize = 8)

	#bbox_to_anchor =(x0, y0, width, height)
	ax4.legend(loc = "center", fontsize = 11, bbox_to_anchor = (0, 2.92, 1.1, 0.2), ncol = m_hrm+1, frameon = False, labelspacing=0.1, handlelength=1, handletextpad=0.3,columnspacing=0.8)
	ax4.grid(visible = True, which = "major", axis = "both", color='w', linestyle='-', linewidth=0.5, zorder = 1, alpha = 0.5)

	plt.savefig("%sfigures/kin_%s_disp_%s.png"%(out,vmode,galaxy))#, transparent=True)
	plt.clf()





