import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
from matplotlib.offsetbox import AnchoredText
from .axes_params import axes_ambient as axs
from .cbar import colorbar as cb
from .colormaps_CLC import vel_map
from .constants import __c__

#params =   {'text.usetex' : True }
#plt.rcParams.update(params)

def vmin_vmax(data,pmin=2,pmax=99.5,base=None,symmetric =False):
	vmin,vmax=np.nanpercentile(np.unique(data),pmin),np.nanpercentile(np.unique(data),pmax)
	vsym = (vmax+abs(vmin))*0.5
	if symmetric: vmin,vmax=-1*vsym,vsym
	if base is not None:
		vmin,vmax=(vmin//base+1)*base,(vmax//base+1)*base
		if symmetric: vmin,vmax=-1*(vsym//base+1)*base,(vsym//base+1)*base
	return vmin,vmax

def zero2nan(data):
	data[data==0]=np.nan
	return data

cmap = vel_map()

def plot_kin_models(galaxy,vmode,momms_mdls,R,Sigma,eSigma,Vrot,eVrot,Vrad,eVrad,Vtan,eVtan,VSYS,INC,ext,hdr_info,config,out):
	_,_,mom0_mdl,mom1_mdl,mom2_mdl_kms,mom2_mdl_A,cube_mdl,velmap_intr,sigmap_intr,twoDmodels= momms_mdls
	vt2D=twoDmodels[0]

	width, height = 9, 6 # width [cm]
	#width, height = 3, 15 # width [cm]
	cm_to_inch = 0.393701 # [inch/cm]
	figWidth = width * cm_to_inch # width [inch]
	figHeight = height * cm_to_inch # width [inch]

	fig, ax2 = plt.subplots(figsize=(6.5, 6.5*0.75), dpi = 300)


	#ax2=plt.subplot(gs2[0,3])

	# is it the axes in arcsec or arcmin ?
	rnorm=1
	if np.max(R)>80:
		rnorm=60
		rlabel='$\'$'
	else:
		rlabel='$\'\'$'

	ext = ext/rnorm
	R=R/rnorm
	axs(ax2, rotation='horizontal',fontsize_ticklabels=20)

	#txt = AnchoredText('$\mathrm{V}_{t}/\sigma=%s$'%(mean_rat),loc='upper left', pad=0.1, borderpad=0, prop=dict(size=10), frameon=0);ax2.add_artist(txt)

	ax2.errorbar(R, Sigma, yerr=eSigma, color = "#db6d52", label = "$\sigma_{intrin}$",  fmt='D', mfc = '#db6d52', mec = '#170a06',ms = 4,mew = 0.5, ecolor='#db6d52', lw=1, ls = ':', capsize=2)
	ax2.errorbar(R, Vrot, yerr=eVrot, color = "#362a1b", label = "$\mathrm{V_{t}}$",  fmt='D', mfc = '#362a1b', mec = '#170a06',ms = 4,mew = 0.5, ecolor='#362a1b', lw=1, ls = ':', capsize=2)


	if vmode == "radial":
		ax2.errorbar(R, Vrad, yerr=eVrad, color = "#c73412", label = "$\mathrm{V_{r}}$", fmt='D', mfc = '#c73412', mec = '#170a06',ms = 4,mew = 0.5, ecolor='#c73412', lw=1, ls = ':', capsize=2)

	if vmode == "vertical":
		ax2.errorbar(R, Vrad, yerr=eVrad, color = "#b47b50", label = "$\mathrm{V_{z}}$", fmt='D', mfc = '#b47b50', mec = '#170a06',ms = 4,mew = 0.5, ecolor='#b47b50', lw=1, ls = ':', capsize=2)

	if vmode == "bisymmetric":
		ax2.errorbar(R, Vrad, yerr=eVrad, color = "#c73412", label = "$\mathrm{V_{2,r}}$", fmt='D', mfc = '#c73412', mec = '#170a06',ms = 4,mew = 0.5, ecolor='#c73412', lw=1, ls = ':', capsize=2)
		ax2.errorbar(R, Vtan, yerr=eVtan, color = "#2fa7ce", label = "$\mathrm{V_{2,t}}$", fmt='D', mfc = '#2fa7ce', mec = '#170a06',ms = 4,mew = 0.5, ecolor='#2fa7ce', lw=1, ls = ':', capsize=2)

	#bbox_to_anchor =(x0, y0, width, height)
	ax2.legend(loc = "center", fontsize = 20, bbox_to_anchor = (0, 1, 1, 0.2), ncol = 4, frameon = False, labelspacing=0.1, handlelength=1, handletextpad=0.3,columnspacing=0.8)

	vels0 = np.asarray([0*Vrot, Vrot, Vrad, Vtan])
	vels=vels0.flatten()
	msk=(np.isfinite(vels))
	vels=vels[msk]
	max_vel,min_vel = int(np.nanmax(vels)),int(np.nanmin(vels))
	min_vel = abs(min_vel)
	pad=40

	M=25
	if max_vel> 120:
		M=30
	if max_vel>190:
		M=40
	if max_vel>250:
		M=50
	if max_vel>300:
		M=60
	if max_vel < 50:
		M = 10 if max_vel > 10 else 1
		ax2.set_ylim(-1*(min_vel//1)-1,1.25*(max_vel//1))
	else:
		ax2.set_ylim(-30*(min_vel//30)-30,30*(max_vel//30)+40)
	ax2.yaxis.set_major_locator(MultipleLocator(M))

	ax2.plot([0,np.nanmax(R)],[0,0],color = "k",linestyle='-', alpha = 0.6,linewidth = 0.3)
	ax2.set_xlabel(f'r ({rlabel})',fontsize=20)
	ax2.set_ylabel('$\mathrm{V_{rot}~(km~s^{-1})}$',fontsize=20)


	"""
	# plot PSF ellipse ?
	config_general = config['general']
	eline=config_general.getfloat('eline')
	specres=config_general.getfloat('fwhm_inst',None)

	if specres	is not None:
		specres_kms=(specres/eline)*__c__
		fwhm_kms=specres_kms/2.354
	else:
		fwhm_kms=hdr_info.cdelt3_kms

	fwhm_kms=fwhm_kms*np.sin(INC*np.pi/180)
	x0,y0=np.nanmax(R)*0.05, max_vel*0.8
	ax2.errorbar(x0, y0, yerr=fwhm_kms,fmt ='s',lw=0.5,color='k', ms=1)
    """
	fig.tight_layout()
	plt.savefig("%sfigures/kin_%s_disp_%s.png"%(out,vmode,galaxy))
	#plt.clf()
	plt.close()
