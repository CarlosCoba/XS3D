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


#params =   {'text.usetex' : True }
#plt.rcParams.update(params) 

def vmin_vmax(data,pmin=5,pmax=98,base=None):
	vmin,vmax=np.nanpercentile(np.unique(data),pmin),np.nanpercentile(np.unique(data),pmax)
	if base is not None:
		vmin,vmax=(vmin//base+1)*base,(vmax//base+1)*base
	return vmin,vmax

def zero2nan(data):
	data[data==0]=np.nan
	return data

cmap = vel_map()

def plot_kin_models(galaxy,vmode,momms_mdls,R,Sigma,eSigma,Vrot,eVrot,Vrad,eVrad,Vtan,eVtan,VSYS,ext,out):
	mom0_mdl,mom1_mdl,mom2_mdl_kms,mom2_mdl_A,cube_mdl,velmap_intr,sigmap_intr= momms_mdls


	width, height = 14.0, 5.5 # width [cm]
	#width, height = 3, 15 # width [cm]
	cm_to_inch = 0.393701 # [inch/cm]
	figWidth = width * cm_to_inch # width [inch]
	figHeight = height * cm_to_inch # width [inch]
  
	fig = plt.figure(figsize=(figWidth, figHeight), dpi = 300)
	#nrows x ncols
	widths = [1, 1, 0.3,1.5]
	heights = [1]
	gs2 = gridspec.GridSpec(1, 4,  width_ratios=widths, height_ratios=heights)
	gs2.update(left=0.06, right=0.99,top=0.81,bottom=0.14, hspace = 0.03, wspace = 0)

	ax0=plt.subplot(gs2[0,0])
	ax1=plt.subplot(gs2[0,1])
	ax2=plt.subplot(gs2[0,3])



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

	Vrad[Vrad == 0] = np.nan
	Vtan[Vtan == 0] = np.nan



	axs(ax0)
	axs(ax1,remove_yticks= True)
	axs(ax2)



	ax0.set_ylabel('$\mathrm{ \Delta Dec~(arcsec)}$',fontsize=10,labelpad=1)
	ax0.set_xlabel('$\mathrm{ \Delta RA~(arcsec)}$',fontsize=10,labelpad=1)
	ax1.set_xlabel('$\mathrm{ \Delta RA~(arcsec)}$',fontsize=10,labelpad=1)


	txt = AnchoredText('$\mathrm{V}_{intrinsic}$', loc="upper left", pad=0.1, borderpad=0, prop={"fontsize":11},zorder=1e4);ax0.add_artist(txt)
	txt = AnchoredText('$\sigma_{intrinsic}$', loc="upper left", pad=0.1, borderpad=0, prop={"fontsize":11},zorder=1e4);ax1.add_artist(txt)


	ax2.plot(R,Sigma, color = "gold",linestyle='--', alpha = 1, linewidth=0.8, label = "$\sigma_{intrin}$")
	ax2.fill_between(R, Sigma-eSigma, Sigma+eSigma, color = "gold", alpha = 0.3, linewidth = 0)


	if vmode == "circular":

		ax2.plot(R,Vrot, color = "#362a1b",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{V_{t}}$")
		ax2.fill_between(R, Vrot-eVrot, Vrot+eVrot, color = "#362a1b", alpha = 0.3, linewidth = 0)

	if vmode == "radial":

		ax2.plot(R,Vrot, color = "#362a1b",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{V_{t}}$")
		ax2.fill_between(R, Vrot-eVrot, Vrot+eVrot, color = "#362a1b", alpha = 0.3, linewidth = 0)

		ax2.plot(R,Vrad, color = "#c73412",linestyle='-', alpha = 0.6, linewidth=0.8, label = "$\mathrm{V_{r}}$")
		ax2.fill_between(R, Vrad-eVrad, Vrad+eVrad, color = "#c73412", alpha = 0.3, linewidth = 0)

	if vmode == "bisymmetric":

		ax2.plot(R,Vrot, color = "#362a1b",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{V_{t}}$")
		ax2.fill_between(R, Vrot-eVrot, Vrot+eVrot, color = "#362a1b", alpha = 0.3, linewidth = 0)

		ax2.plot(R,Vrad, color = "#c73412",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{V_{2,r}}$")
		ax2.fill_between(R, Vrad-eVrad, Vrad+eVrad, color = "#c73412", alpha = 0.3, linewidth = 0)

		ax2.plot(R,Vtan, color = "#2fa7ce",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{V_{2,t}}$")
		ax2.fill_between(R, Vtan-eVtan, Vtan+eVtan, color = "#2fa7ce", alpha = 0.3, linewidth = 0)




	#bbox_to_anchor =(x0, y0, width, height)
	ax2.legend(loc = "center", fontsize = 10, bbox_to_anchor = (0, 1, 1, 0.2), ncol = 4, frameon = False, labelspacing=0.1, handlelength=1, handletextpad=0.3,columnspacing=0.8)

	vels0 = np.asarray([0*Vrot, Vrot, Vrad, Vtan])
	vels=vels0.flatten()
	msk=(np.isfinite(vels)) & (vels!=0)
	vels=vels[msk]
	max_vel,min_vel = int(np.nanmax(vels)),int(np.nanmin(vels))
	min_vel = abs(min_vel)

	ax2.set_ylim(-50*(min_vel//50)-50,50*(max_vel//50)+80)
	ax2.plot([0,np.nanmax(R)],[0,0],color = "k",linestyle='-', alpha = 0.6,linewidth = 0.3)
	ax2.set_xlabel('$\mathrm{r~(arcsec)}$',fontsize=10,labelpad = 2)
	ax2.set_ylabel('$\mathrm{V_{rot}~(km~s^{-1})}$',fontsize=10,labelpad = 2)

	cb(im0,ax0,orientation = "horizontal", colormap = cmap, bbox= (0.10,1.1,0.8,1),width = "100%", height = "5%",label_pad = -26, label = "$\mathrm{(km~s^{-1})}$",labelsize=10, ticksfontsize = 8)
	cb(im1,ax1,orientation = "horizontal", colormap = cmap, bbox= (0.10,1.1,0.8,1),width = "100%", height = "5%",label_pad = -26, label = "$\mathrm{(km~s^{-1})}$",labelsize=10, ticksfontsize = 8)
	ax2.grid(visible = True, which = "major", axis = "both", color='w', linestyle='-', linewidth=0.5, zorder = 1, alpha = 0.5)

	plt.savefig("%sfigures/kin_%s_disp_%s.png"%(out,vmode,galaxy))
	#plt.clf()
	plt.close()	




