import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker

def colorbar(im,axis,orientation="vertical",labelsize=10,colormap="rainbow",ticks='None',label=None,bbox=(1, 1, 1, 1),width="100%",height="100%",label_pad=0, ticksfontsize = 12, colors  = "k", pad = 1, power=False):
	cax1 = inset_axes(axis,
		width=width,
		height=height, 
		loc=3,
		bbox_to_anchor=bbox,
		bbox_transform=axis.transAxes,
		borderpad=0
		)

	def fmt(x0, pos):
		x = float(x0)		
		check = x.is_integer()
		if check :
			return "%s"%int(x)
		else:
			return "%0.1f"%x

	if not power:
		kwargs={'format':ticker.FuncFormatter(fmt)}
	else:
		kwargs={}	
	
	if ticks == 'None':
		cbar1=plt.colorbar(im,cax=cax1,orientation=orientation,cmap=colormap, **kwargs)
	else:
		cbar1=plt.colorbar(im,cax=cax1,orientation=orientation,cmap=colormap,ticks=ticks, **kwargs)

	if label != None:
		if orientation == "vertical":
			rot = 90
			rot_ticks = 90
		else:
			rot = 0
			rot_ticks=0
		cbar1.set_label(fontsize=labelsize,label=label,rotation=rot,labelpad=label_pad, color = colors)
		#cbar1.ax.get_yaxis().labelpad = label_pad

	cbar1.ax.tick_params(color = colors)
	cbar1.ax.tick_params(axis='y', direction='in',rotation=rot_ticks, labelsize=ticksfontsize)
	cax1.tick_params(direction='in', pad = pad, width = 0.5)
	cax1.tick_params(colors=colors, which='both') 

	cbar1.outline.set_edgecolor(colors)
	cbar1.outline.set_linewidth(0.2)

	# scientific notation
	if power:
		cbar1.formatter.set_powerlimits((0, 0))
	
	return cax1


