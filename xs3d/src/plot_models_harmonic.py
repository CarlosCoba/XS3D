import numpy as np
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

from matplotlib.offsetbox import AnchoredText
from .axes_params import axes_ambient as axs
from .cbar import colorbar as cb
from .colormaps_CLC import vel_map


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
	mom0_mdl,mom1_mdl,mom2_mdl_kms,mom2_mdl_A,cube_mdl,velmap_intr,sigmap_intr,twoDmodels= momms_mdls
	c1 = Ck[0]
	e_c1 = e_Ck[0]
	s_1 = Sk[0]
	Rnoncirc = R*(s_1/s_1)


	width, height = 14.0, 5.5 # width [cm]
	#width, height = 3, 15 # width [cm]
	cm_to_inch = 0.393701 # [inch/cm]
	figWidth = width * cm_to_inch # width [inch]
	figHeight = height * cm_to_inch # width [inch]

	fig = plt.figure(figsize=(figWidth, figHeight), dpi = 300)
	#nrows x ncols
	widths = [1, 1, 0.4,1.5]
	heights = [1, 0.6]
	gs = gridspec.GridSpec(2, 4,  width_ratios=widths, height_ratios=heights)
	gs.update(left=0.06, right=0.99,top=0.81,bottom=0.14, hspace = 0.03, wspace = 0)

	ax0 = plt.subplot(gs[0:2,0])
	ax1 = plt.subplot(gs[0:2,1])

	ax3 = plt.subplot(gs[0,3])
	ax4 = plt.subplot(gs[1,3])

	# is it the axes in arcsec or arcmin ?
	rnorm=1
	if np.max(R)>80:
		rnorm=60
		rlabel='$\'$'
	else:
		rlabel='$\'\'$'
	ext = ext/rnorm
	R=R/rnorm
	Rnoncirc=Rnoncirc/rnorm

	# intrinsic velocity
	vt=zero2nan(twoDmodels[0])
	velmax = 10*(np.nanmax(vt) // 10)

	im0=ax0.imshow(vt,origin='lower',cmap=cmap,extent=ext,vmin=0,vmax=velmax,aspect='auto')


	# intrinsic velocity dispersion
	sigmap_intr=zero2nan(sigmap_intr)
	vmin = abs(np.nanmin(sigmap_intr))
	vmax = abs(np.nanmax(sigmap_intr))
	max_vel = np.nanmax([vmin,vmax])
	vmin = -(max_vel//25 + 1)*25
	vmax = (max_vel//25 + 1)*25
	vmin,vmax=vmin_vmax(sigmap_intr,base=10)
	im1=ax1.imshow(sigmap_intr,origin='lower',cmap=cmap,extent=ext,vmin=vmin,vmax=vmax,aspect='auto')


	axs(ax0)
	axs(ax1,remove_yticks= True)


	ax0.set_ylabel('$\mathrm{ \Delta Dec}$ (%s)'%rlabel,fontsize=10,labelpad=1)
	ax0.set_xlabel('$\mathrm{ \Delta RA}$ (%s)'%rlabel,fontsize=10,labelpad=1)
	ax1.set_xlabel('$\mathrm{ \Delta RA}$ (%s)'%rlabel,fontsize=10,labelpad=1)


	txt = AnchoredText('$\mathrm{V}_{intrinsic}$', loc="upper left", pad=0.1, borderpad=0, prop={"fontsize":11},zorder=1e4);ax0.add_artist(txt)
	txt = AnchoredText('$\sigma_{intrinsic}$', loc="upper left", pad=0.1, borderpad=0, prop={"fontsize":11},zorder=1e4);ax1.add_artist(txt)

	ax3.plot(R,c1, color = "#362a1b",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{c_{1}}$")
	ax3.fill_between(R, c1-e_c1, c1+e_c1, color = "#362a1b", alpha = 0.3, linewidth = 0)
	ax3.scatter(R,c1,s=20,marker='s',c='#362a1b',edgecolor='k',lw=0.3,zorder=10)

	ax3.plot(R,Sigma, color = "#db6d52",linestyle='--', alpha = 1, linewidth=0.8, label = "$\sigma_{intr}$")
	ax3.fill_between(R, Sigma-eSigma, Sigma+eSigma, color = "#db6d52", alpha = 0.3, linewidth = 0)
	ax3.scatter(R,Sigma,s=20,marker='s',c='#db6d52',edgecolor='k',lw=0.3,zorder=10)

	ax3.set_ylim(0, 30*(np.max(c1)//30)+40)
	ax3.set_xlim(0, np.max(R))
	axs(ax3, remove_xticks= True, rotation = 'horizontal')


	# plot the 1D velocities with the predefined colors
	ax4.plot(R,-1e4*Sigma, color = "#db6d52",linestyle='--', alpha = 1, linewidth=0.8, label = "$\sigma_{intr}$")
	ax4.plot(R,-1e4*np.ones(len(c1)), color = "#362a1b",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{c_{1}}$")
	ax4.plot([0,np.nanmax(R)],[0,0],color = "k",linestyle='-', alpha = 0.6,linewidth = 0.3)
	if 2*m_hrm < len(list_fancy_colors):
		for i in range(m_hrm):
			color = list_fancy_colors
			if i >= 1:
				ax4.plot(Rnoncirc,Ck[i], color = color[i],linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{c_{%s}}$"%(i+1))
				ax4.fill_between(Rnoncirc, Ck[i]-e_Ck[i], Ck[i]+e_Ck[i], color = color[i], alpha = 0.3, linewidth = 0)
				ax4.scatter(Rnoncirc,Ck[i],s=4,marker='s',c=color[i],edgecolor='k',lw=0.3,zorder=10)
			ax4.plot(Rnoncirc,Sk[i], color = color[i+m_hrm],linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{s_{%s}}$"%(i+1))
			ax4.fill_between(Rnoncirc, Sk[i]-e_Sk[i], Sk[i]+e_Sk[i], color = color[i+m_hrm], alpha = 0.3, linewidth = 0)
			ax4.scatter(Rnoncirc,Sk[i],s=4,marker='s',c=color[i+m_hrm],edgecolor='k',lw=0.3,zorder=10)
	else:
		ax4.clear()
		# pick a random color
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
				ax4.scatter(Rnoncirc,Ck[i],s=4,marker='s',c=colors[k1],edgecolor='k',lw=0.3,zorder=10)

			ax4.plot(Rnoncirc,Sk[i], color = colors[k2],linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{s_{%s}}$"%(i+1))
			ax4.fill_between(Rnoncirc, Sk[i]-e_Sk[i], Sk[i]+e_Sk[i], color = colors[k2], alpha = 0.3, linewidth = 0)
			ax4.scatter(Rnoncirc,Sk[i],s=4,marker='s',c=colors[k2],edgecolor='k',lw=0.3,zorder=10)

	axs(ax4,  rotation = 'horizontal')
	vmin_s1 = 5*abs(np.nanmin(Sk[0]))//5
	vmax_s1 = 5*abs(np.nanmax(Sk[0]))//5
	max_vel_s1 = np.nanmax([vmin_s1,vmax_s1])
	ax4.set_ylim(-max_vel_s1 -4 , max_vel_s1 + 4)
	ax4.set_xlim(0, np.max(R))
	ax4.set_xlabel(f"r ({rlabel})", labelpad = 2, fontsize = 10)
	ax3.set_ylabel("$\mathrm{c_{1} (km/s)}$",fontsize = 10,labelpad=2)
	ax4.set_ylabel("$\mathrm{v_{non-circ}}$ \n $\mathrm{(km/s)}$",fontsize = 10,labelpad=2)


	cb(im0,ax0,orientation = "horizontal", colormap = cmap, bbox=(0.10,1.1,0.8,1),width = "100%", height = "5%",label_pad = -24, label = "$\mathrm{(km~s^{-1})}$",labelsize=10, ticksfontsize = 8)
	cb(im1,ax1,orientation = "horizontal", colormap = cmap, bbox=(0.10,1.1,0.8,1),width = "100%", height = "5%",label_pad = -24, label = "$\mathrm{(km~s^{-1})}$",labelsize=10, ticksfontsize = 8)

	#bbox_to_anchor =(x0, y0, width, height)
	ax4.legend(loc = "center", fontsize = 11, bbox_to_anchor = (0, 2.92, 1.1, 0.2), ncol = m_hrm+1, frameon = False, labelspacing=0.1, handlelength=1, handletextpad=0.3,columnspacing=0.8)
	ax4.grid(visible = True, which = "major", axis = "both", color='w', linestyle='-', linewidth=0.5, zorder = 1, alpha = 0.5)
	plt.savefig("%sfigures/kin_%s_disp_%s.png"%(out,vmode,galaxy))#, transparent=True)
	plt.clf()
