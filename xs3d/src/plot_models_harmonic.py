import numpy as np
import matplotlib
import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
import matplotlib.ticker as ticker
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

def inter_from_256(x):
	return np.interp(x=x,xp=[0,255],fp=[0,1])
cdict = {
	'red':((0.0,inter_from_256(64),inter_from_256(64)),
		   (1/5*1,inter_from_256(112),inter_from_256(112)),
		   (1/5*2,inter_from_256(230),inter_from_256(230)),
		   (1/5*3,inter_from_256(253),inter_from_256(253)),
		   (1/5*4,inter_from_256(244),inter_from_256(244)),
		   (1.0,inter_from_256(169),inter_from_256(169))),
	'green': ((0.0, inter_from_256(57), inter_from_256(57)),
			(1 / 5 * 1, inter_from_256(198), inter_from_256(198)),
			(1 / 5 * 2, inter_from_256(241), inter_from_256(241)),
			(1 / 5 * 3, inter_from_256(219), inter_from_256(219)),
			(1 / 5 * 4, inter_from_256(109), inter_from_256(109)),
			(1.0, inter_from_256(23), inter_from_256(23))),
	'blue': ((0.0, inter_from_256(144), inter_from_256(144)),
			  (1 / 5 * 1, inter_from_256(162), inter_from_256(162)),
			  (1 / 5 * 2, inter_from_256(246), inter_from_256(146)),
			  (1 / 5 * 3, inter_from_256(127), inter_from_256(127)),
			  (1 / 5 * 4, inter_from_256(69), inter_from_256(69)),
			  (1.0, inter_from_256(69), inter_from_256(69))),
}
from matplotlib import colors
new_cmap = colors.LinearSegmentedColormap('new_cmap',segmentdata=cdict)

def plot_kin_models_h(galaxy,vmode,best_vals,best_vels,result, m_hrm, out):

	scalar_fields = ["v_disp"]
	vels = {k:best_vals[k] for k in scalar_fields}
	vdisp = vels['v_disp']		
	evdisp=np.zeros_like(vdisp)
	
	R=abs(best_vals['radius'])
	nrings=len(R)	

	Sk=[]
	Ck=[]
	
	e_Sk=[]
	e_Ck=[]
	
	eNC = [e_Ck,e_Sk]
	evdisp = []	
	for m in range(m_hrm):
		k = str(int(m+1))
		c_k = best_vels[f'c_m{k}']
		s_k = best_vels[f's_m{k}']
		Ck.append(c_k)		
		Sk.append(s_k)				

		vnc = [f'c_m{k}_r',f's_m{k}_r']
		for w,v in enumerate(vnc):
			vtmp=[]
			for n in range(nrings):
				try:
					error=result.params[v+f'{n}'].stderr
					vtmp.append(error if error is not None else 0)
				except(KeyError):
					vtmp.append(0)	
			(eNC[w]).append(vtmp)

	#  dispersion
	for n in range(nrings):
		try:
			error=result.params['v_disp_r'+f'{n}'].stderr
			evdisp.append(error if error is not None else 0)
		except(KeyError):
			evdisp.append(0)	
			

	[e_Ck,e_Sk] = eNC
	c1 = Ck[0]
	e_c1 = e_Ck[0]
	s_1 = Sk[0]
	R_noncirc = R*(s_1/s_1)

	width, height = 11, 9 # width [cm]
	#width, height = 3, 15 # width [cm]
	cm_to_inch = 0.393701 # [inch/cm]
	figWidth = width * cm_to_inch # width [inch]
	figHeight = height * cm_to_inch # width [inch]

	fig = plt.figure(figsize=(figWidth, figHeight), dpi = 300)
	#nrows x ncols
	widths = [1, 1, 0.4,1.5]
	heights = [1, 0.6]
	gs = gridspec.GridSpec(2, 1, height_ratios=heights)
	gs.update(left=0.15, right=0.85,top=0.97,bottom=0.13, hspace = 0.05, wspace = 0)


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
	r_nc=R_noncirc/rnorm

	delta_r = R[-1]-R[-2]
	max_r = np.max(R)

	ax0.errorbar(R,c1,yerr=e_c1,color="#362a1b",label = "$\mathrm{c_{1}}$",fmt='o',
		mfc='#362a1b',mec='#362a1b',ms=4,mew=0.5,ecolor='#362a1b',lw=1,ls='-',capsize=2)	

	ax0.errorbar(R,vdisp,yerr=evdisp,color='#db6d52',label="$\sigma_{gas}$",fmt='o',
		mfc='#db6d52',mec='#db6d52',ms=4,mew=0.5,ecolor='#db6d52',lw=1,ls='--',capsize=2)	


	ax0.set_ylim(0, 30*(np.max(c1)//30)+40)
	ax0.set_xlim(-delta_r*0.1, max_r+delta_r*0.1)
	axs(ax0, remove_xticks= True, rotation = 'horizontal', fontsize_ticklabels=10)


	# plot the 1D velocities with the predefined colors
	ax1.plot(R,-1e4*np.ones(len(c1)), color = "#db6d52",linestyle='--', alpha = 1, linewidth=0.8, label = "$\sigma_\mathrm{gas}$")
	ax1.plot(R,-1e4*np.ones(len(c1)), color = "#362a1b",linestyle='-', alpha = 1, linewidth=0.8, label = "$\mathrm{c_{1}}$")
	ax1.plot([0,np.nanmax(R)],[0,0],color = "k",linestyle='-', alpha = 0.6,linewidth = 0.3)

	for i in range(m_hrm):
			m = int(2*m_hrm)
			color = new_cmap(np.linspace(0, 1, m))
			if i >= 1:
				ax1.errorbar(r_nc,Ck[i],yerr=e_Ck[i],color=color[i],label="$\mathrm{c_{%s}}$"%(i+1),fmt='o',
					mfc=color[i],mec=color[i],ms=4,mew=0.5,ecolor=color[i],lw=1,ls='-',capsize=2)	
			
			ax1.errorbar(r_nc,Sk[i],yerr=e_Sk[i],color=color[i+m_hrm],label="$\mathrm{s_{%s}}$"%(i+1),fmt='o',
					mfc=color[i+m_hrm],mec=color[i+m_hrm],ms=4,mew=0.5,ecolor=color[i+m_hrm],lw=1,ls='-',capsize=2)	

			
	axs(ax1,  rotation = 'horizontal', fontsize_ticklabels=10)
	vmin_s1 = 5*abs(np.nanmin(Sk[0]))//5
	vmax_s1 = 5*abs(np.nanmax(Sk[0]))//5
	max_vel_s1 = np.nanmax([vmin_s1,vmax_s1])
	
	dashline = (5, (10, 3))
	ax1.plot([0,max_r],[0,0],color = "k",linestyle=dashline, alpha = 1,linewidth = 0.5)						
	ax1.set_ylim(-max_vel_s1 -4 , max_vel_s1 + 4)
	ax1.set_xlabel(f"r ({rlabel})", fontsize = 10)
	ax1.set_xlim(-delta_r*0.1, max_r+delta_r*0.1)	
	ax0.set_ylabel("$\mathrm{Velocity~(km\,s^{-1})}$",fontsize = 10)
	ax1.set_ylabel("$\mathrm{V_{NC}~(km\,s^{-1})}$",fontsize = 10)

	ax0.yaxis.set_major_locator(ticker.MaxNLocator(5))
	ax0.xaxis.set_major_locator(ticker.MaxNLocator(5))
	ax1.xaxis.set_major_locator(ticker.MaxNLocator(5))		
	ax1.yaxis.set_major_locator(ticker.MaxNLocator(3))
	
	ax1.legend(loc="upper left", fontsize=10, bbox_to_anchor=(1,1), ncol=1, bbox_transform=ax0.transAxes, labelspacing=0.1, handlelength=1, handletextpad=0.3, columnspacing=0.8, frameon=False)
		
	ax1.grid(visible = True, which = "major", axis = "both", color='gray', linestyle='-', linewidth=0.5, zorder = 1, alpha = 0.5)

	#fig.tight_layout()
	plt.savefig("%sfigures/kin_%s_disp_%s.png"%(out,vmode,galaxy))#, transparent=True)
	plt.clf()
