import numpy as np
import matplotlib.pylab as plt

from matplotlib.ticker import FormatStrFormatter
majorFormatter = FormatStrFormatter("$%g$")

#def axes_ambient(axis,xlabel=None,ylabel=None,remove_xticks= False,remove_yticks= False, remove_xyticks=False, remove_ticks_all = False, tickscolor = "k", fontsize_xticklabel = 10, fontsize_yticklabel = 10, fontsize_ticklabels = None,  rotation = "vertical", frame = False, remove_axis_lines = 0 ):

def axes_ambient(axis,xlabel=None,ylabel=None,remove_xticks= False,remove_yticks= False, remove_xyticks=False, remove_ticks_all = False, tickscolor = "k", fontsize_xticklabel = 10, fontsize_yticklabel = 10, fontsize_ticklabels = 8,  rotation = "vertical", frame = False, remove_axis_lines = 0, direction='in' ):
	
	fontsize_xticklabel, fontsize_yticklabel = fontsize_ticklabels, fontsize_ticklabels
	plt.setp(axis.get_yticklabels(), rotation=rotation, fontsize=fontsize_yticklabel)#,visible=False)
	plt.setp(axis.get_xticklabels(), fontsize=fontsize_xticklabel)



	if remove_axis_lines == False:
		axis.spines['bottom'].set_color(tickscolor)
		axis.spines['top'].set_color(tickscolor)
		axis.spines['left'].set_color(tickscolor)
		axis.spines['right'].set_color(tickscolor)
		axis.minorticks_on()
		axis.tick_params('both', length=6.5, width=0.3, which='major',direction=direction,color=tickscolor,bottom=1, top=1, left=1, right=1)
		axis.tick_params('both', length=3.5, width=0.3, which='minor',direction=direction,color=tickscolor,bottom=1, top=1, left=1, right=1) 

	else:

		axis.spines['right'].set_color('none')
		axis.spines['top'].set_color('none')
		axis.spines['left'].set_position(('data',0))
		axis.spines["bottom"].set_position(("data",0))

		axis.minorticks_on()
		axis.tick_params('x', length=6.5, width=0.3, which='major',direction=direction,color=tickscolor,top=0)
		axis.tick_params('x', length=3.5, width=0.3, which='minor',direction=direction,color=tickscolor,top=0)
 
		axis.tick_params('y', length=6.5, width=0.3, which='major',direction=direction,color=tickscolor,right=0)
		axis.tick_params('y', length=3.5, width=0.3, which='minor',direction=direction,color=tickscolor,right=0) 

		plt.setp(axis.get_yticklabels(), rotation=rotation, fontsize=fontsize_yticklabel)#,visible=False)
		plt.setp(axis.get_xticklabels(), fontsize=fontsize_xticklabel)


	axis.tick_params(axis='x', colors='k',pad=3)
	axis.tick_params(axis='y', colors='k',pad=3)
	axis.tick_params(axis='both',direction=direction,color=tickscolor)


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
	if remove_xticks:
		axis.xaxis.set_major_formatter(plt.NullFormatter())

	if remove_yticks:
		axis.yaxis.set_major_formatter(plt.NullFormatter())

	if remove_xyticks:
		axis.yaxis.set_major_formatter(plt.NullFormatter())
		axis.xaxis.set_major_formatter(plt.NullFormatter())		

	# remove all from the axis (both ticks and xy ticks labels)
	if remove_ticks_all == True:
		axis.yaxis.set_major_locator(plt.NullLocator())
		axis.xaxis.set_major_locator(plt.NullLocator())



	# remove frame
	if frame == True:
		axis.axis('off')




	# if remove x,y lines
	#axis.set_facecolor(facecolor)


	# Intead of padding in point units, you can pad with axes units: 
	#ax.yaxis.set_label_coords(-0.15, 0.5)

