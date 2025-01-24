import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec


def fig_ambient(fig_size = (2,2), ncol = 2, nrow = 2, left = 0.15, right = 0.95, top = 0.95, hspace = 0, bottom = 0.15, wspace = 0 ):

	fig=plt.figure(figsize= fig_size )
	gs2 = GridSpec(nrow, ncol)
	gs2.update(left=left, right=right,top=top,hspace=hspace,bottom=bottom,wspace=wspace)


	nplots = ncol*nrow
	axes = []
	for k in range(nplots):
		ax = plt.subplot(gs2[k])
		#AXIS(ax)
		axes.append(ax)
	

	return axes


