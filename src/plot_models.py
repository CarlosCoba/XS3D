import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
from matplotlib.offsetbox import AnchoredText
from src.axes_params import axes_ambient as axs 
from src.cbar import colorbar as cb
from src.colormaps_CLC import vel_map
from src.constants import __c__

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
	mom0_mdl,mom1_mdl,mom2_mdl_kms,mom2_mdl_A,cube_mdl,velmap_intr,sigmap_intr,twoDmodels= momms_mdls
	vt2D=twoDmodels[0]

	width, height = 9, 6 # width [cm]
	#width, height = 3, 15 # width [cm]
	cm_to_inch = 0.393701 # [inch/cm]
	figWidth = width * cm_to_inch # width [inch]
	figHeight = height * cm_to_inch # width [inch]
  
	fig = plt.figure(figsize=(figWidth, figHeight), dpi = 300)
	gs1 = fig.add_gridspec(nrows=2, ncols=1, left=0.8, right=0.98, top=0.87, bottom=0.13, hspace=0.7)
	gs2 = fig.add_gridspec(nrows=1, ncols=1, left=0.13, right=0.7, wspace=0.2, bottom=0.14)

	ax0=plt.subplot(gs1[0,0])
	ax1=plt.subplot(gs1[1,0])
	ax2=plt.subplot(gs2[0,0])
				
	#ax2=plt.subplot(gs2[0,3])

	# is it the axes in arcsec or arcmin ?
	rnorm=1
	if np.max(ext)>80:
		rnorm=60
		rlabel='$\'$'
	else:
		rlabel='$\'\'$'
			
	ext = ext/rnorm
	R=R/rnorm
	
	
	# intrinsic velocity
	vt=zero2nan(vt2D)
	vmax=np.max(vt)
	#velmap=velmap_intr-VSYS
	base=10 if vmax > 50 else 1
	velmax = base*(np.nanmax(vt) // base) 		
	im0=ax0.imshow(vt,origin='lower',cmap=cmap,extent=ext, vmin=0,vmax=velmax, aspect='auto')
	
	
	# intrinsic velocity dispersion
	sigmap_intr=zero2nan(sigmap_intr)
	vmin = abs(np.nanmin(sigmap_intr))
	vmax = abs(np.nanmax(sigmap_intr))
	max_vel = np.nanmax([vmin,vmax])	
	vmin = -(max_vel//25 + 1)*25
	vmax = (max_vel//25 + 1)*25
	vmin,vmax=vmin_vmax(sigmap_intr)
	im1=ax1.imshow(sigmap_intr,origin='lower',cmap=cmap,extent=ext,vmin=0,vmax=vmax,aspect='auto')		
	rat=np.divide(vt,sigmap_intr,where=sigmap_intr!=0,out=np.zeros_like(vt))
	mean_rat=np.mean(rat,where = ( (rat!=0) & np.isfinite(rat)) )
	mean_rat=round(mean_rat,0)
	
	Vrad[Vrad == 0] = np.nan
	Vtan[Vtan == 0] = np.nan



	axs(ax0,fontsize_ticklabels=8)
	axs(ax1,fontsize_ticklabels=8)
	axs(ax2, rotation='horizontal',fontsize_ticklabels=8)



	ax0.set_ylabel('$\mathrm{ \Delta Dec}$ (%s)'%rlabel,fontsize=8,labelpad=1)
	#ax0.set_xlabel('$\mathrm{ \Delta RA}$ (%s)'%rlabel,fontsize=8,labelpad=1)
	ax1.set_xlabel('$\mathrm{ \Delta RA}$ (%s)'%rlabel,fontsize=8,labelpad=1)
	ax1.set_ylabel('$\mathrm{ \Delta Dec}$ (%s)'%rlabel,fontsize=8,labelpad=1)

	txt = AnchoredText('$\mathrm{V}_{t}$', loc="upper left", pad=0.1, borderpad=0, prop={"fontsize":11},zorder=1e4);ax0.add_artist(txt)
	txt = AnchoredText('$\sigma$', loc="upper left", pad=0.1, borderpad=0, prop={"fontsize":11},zorder=1e4);ax1.add_artist(txt)
	#txt = AnchoredText('$\mathrm{V}_{t}/\sigma=%s$'%(mean_rat), loc="upper left", pad=0.1, borderpad=0, prop={"fontsize":11},zorder=1e4);ax1.add_artist(txt)
	txt = AnchoredText('$\mathrm{V}_{t}/\sigma=%s$'%(mean_rat),loc='upper left', pad=0.1, borderpad=0, prop=dict(size=10), frameon=0);ax2.add_artist(txt)
                       	

	ax2.plot(R,Sigma, color = "#db6d52",linestyle='--', alpha = 1, linewidth=0.8, label = "$\sigma_{intrin}$")
	ax2.fill_between(R, Sigma-eSigma, Sigma+eSigma, color = "#db6d52", alpha = 0.3, linewidth = 0)
	ax2.scatter(R,Sigma,s=20,marker='s',c='#db6d52',edgecolor='k',lw=0.3,zorder=10)

	ax2.plot(R,Vrot, color = "#362a1b",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{V_{t}}$")
	ax2.fill_between(R, Vrot-eVrot, Vrot+eVrot, color = "#362a1b", alpha = 0.3, linewidth = 0)
	ax2.scatter(R,Vrot,s=20,marker='s',c='#362a1b',edgecolor='k',lw=0.3,zorder=10)
	
	if vmode == "radial":
		ax2.plot(R,Vrad, color = "#c73412",linestyle='-', alpha = 0.6, linewidth=0.8, label = "$\mathrm{V_{r}}$")
		ax2.fill_between(R, Vrad-eVrad, Vrad+eVrad, color = "#c73412", alpha = 0.3, linewidth = 0)
		ax2.scatter(R,Vrad,s=20,marker='s',c='#c73412',edgecolor='k',lw=0.3,zorder=10)

	if vmode == "vertical":
		ax2.plot(R,Vrad, color = "#b47b50",linestyle='-', alpha = 0.6, linewidth=0.8, label = "$\mathrm{V_{z}}$")
		ax2.fill_between(R, Vrad-eVrad, Vrad+eVrad, color = "#b47b50", alpha = 0.3, linewidth = 0)
		ax2.scatter(R,Vrad,s=20,marker='s',c='#b47b50',edgecolor='k',lw=0.3,zorder=10)
			
	if vmode == "bisymmetric":
		ax2.plot(R,Vrad, color = "#c73412",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{V_{2,r}}$")
		ax2.fill_between(R, Vrad-eVrad, Vrad+eVrad, color = "#c73412", alpha = 0.3, linewidth = 0)
		ax2.scatter(R,Vrad,s=20,marker='s',c='#c73412',edgecolor='k',lw=0.3,zorder=10)
		
		ax2.plot(R,Vtan, color = "#2fa7ce",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{V_{2,t}}$")
		ax2.fill_between(R, Vtan-eVtan, Vtan+eVtan, color = "#2fa7ce", alpha = 0.3, linewidth = 0)
		ax2.scatter(R,Vtan,s=20,marker='s',c='#2fa7ce',edgecolor='k',lw=0.3,zorder=10)

	#bbox_to_anchor =(x0, y0, width, height)
	ax2.legend(loc = "center", fontsize = 10, bbox_to_anchor = (0, 1, 1, 0.2), ncol = 4, frameon = False, labelspacing=0.1, handlelength=1, handletextpad=0.3,columnspacing=0.8)

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
	ax2.set_xlabel(f'r ({rlabel})',fontsize=10,labelpad = 2)
	ax2.set_ylabel('$\mathrm{V_{rot}~(km~s^{-1})}$',fontsize=10,labelpad = 0)

	cb(im0,ax0,orientation = "horizontal", colormap = cmap, bbox= (0,1.2,0.65,1),width = "100%", height = "5%",label_pad = -20, label = "$V/\mathrm{km~s^{-1}}$",labelsize=8, ticksfontsize = 8, ticks = [0,velmax])
	cb(im1,ax1,orientation = "horizontal", colormap = cmap, bbox= (0,1.2,0.65,1),width = "100%", height = "5%",label_pad = -20, label = "$\sigma/\mathrm{km~s^{-1}}$",labelsize=8, ticksfontsize = 8)

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
            		
	plt.savefig("%sfigures/kin_%s_disp_%s.png"%(out,vmode,galaxy))
	#plt.clf()
	plt.close()	




