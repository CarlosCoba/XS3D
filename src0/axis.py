import numpy as np
import matplotlib.pylab as plt

from matplotlib.ticker import FormatStrFormatter
majorFormatter = FormatStrFormatter("$%g$")

def AXIS(axis,xlabel=None,ylabel=None,remove_xticks= False,remove_yticks= False,remove_ticks_all = False,tickscolor = "k", fontsize_ticklabel = 8, rotation = "vertical"):#, facecolor = '#e0e0e0' ):
	plt.setp(axis.get_yticklabels(), rotation=rotation, fontsize=fontsize_ticklabel)#,visible=False)
	plt.setp(axis.get_xticklabels(), fontsize=fontsize_ticklabel)

	axis.spines['bottom'].set_color(tickscolor)
	axis.spines['top'].set_color(tickscolor)
	axis.spines['left'].set_color(tickscolor)
	axis.spines['right'].set_color(tickscolor)


	axis.minorticks_on()
	axis.tick_params('both', length=6.5, width=0.3, which='major',direction='in',color=tickscolor,bottom=1, top=1, left=1, right=1)
	axis.tick_params('both', length=3.5, width=0.3, which='minor',direction='in',color=tickscolor,bottom=1, top=1, left=1, right=1) 


	axis.tick_params(axis='x', colors='k',pad=1)
	axis.tick_params(axis='y', colors='k',pad=1)
	axis.tick_params(axis='both',direction='in',color=tickscolor)


	#axis.xaxis.major.locator.set_params(nbins=3) 
	#axis.yaxis.major.locator.set_params(nbins=3) 


	if xlabel != None:
		axis.set_xlabel('%s'%ylabel,fontsize=10)

	if ylabel != None:
		axis.set_xlabel('%s'%xlabel,fontsize=10)



	# Remove unnecessary zeros in ticks
	axis.xaxis.set_major_formatter(majorFormatter) 
	axis.yaxis.set_major_formatter(majorFormatter) 



	#
	# Remove x,y label ticks
	#
	if remove_xticks == True:
		axis.xaxis.set_major_formatter(plt.NullFormatter())


	if remove_yticks == True:
		axis.yaxis.set_major_formatter(plt.NullFormatter())


	# remove all from the axis (both ticks and xy ticks labels)

	if remove_ticks_all == True:
		axis.yaxis.set_major_locator(plt.NullLocator())
		axis.xaxis.set_major_locator(plt.NullLocator())



	#axis.set_facecolor(facecolor)


