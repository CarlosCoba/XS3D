import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

# A nice red and blues
#hex_list = ["#a3dbe6", "#5ea1ba", "#376b94", "#193476", "#01025a",    "#423f46","#531132", "#800d20", "#aa2c24", "#db6d52", "#f1b691"]

# A nice red black and blues
hex_list = ["#01025a", "#193476", "#376b94", "#5ea1ba", "#a3dbe6","#423f46","#f1b691", "#db6d52", "#aa2c24", "#800d20", "#531132"]

blues = ["#01025a", "#193476", "#376b94", "#5ea1ba", "#a3dbe6"]
reds = ["#531132", "#800d20", "#aa2c24", "#db6d52", "#f1b691"]
black = ["#423f46"]


# for intensity maps
mom0 = black+blues[::-1]+reds[::-1]
# for intensity maps
mom0=['#e3e2d1','#a3a08d','#6e655c','#e9d0c8','#dfbcaa','#df6d07','#720c09','#310a34','#616de0','#acbddf','#fafafa']
mom0=['#f9f1f1', '#00002b', '#0e1e76', '#256a7d', '#3e9259', '#90ab55', '#bbaa62', '#d6b197', '#f1dedc', '#f9f1f1']
#mom0=['#e0dfda', '#414c6a', '#637e8f', '#90a6a3', '#c4bca9', '#ba171a','#f8e6c0', '#f2d6a4', '#e4a379', '#996863', '#fefdfd']
#mom0=['#e0dfda', '#414c6a', '#637e8f', '#90a6a3', '#77656a', '#c4bca9','#f8e6c0', '#f2d6a4', '#e4a379', '#996863', '#fefdfd']
mom0=['#bde7db', '#89c4d0', '#54a3cd', '#377dd0', '#4954b1', '#3d3b6c', '#282738', '#1f1e1e', '#3a2126', '#672a39', '#9d2f45', '#cc4239', '#e96e36', '#f5a269', '#ffd4ac']

mom0=['#e0dfda', '#637e8f', '#414c6a', '#1a1a01', '#332312', '#502d1e', '#6f372d', '#8f403b', '#b04b45', '#cd5d4c', '#de754f', '#e58b51', '#eaa053', '#efb759', '#f5d06b', '#fae58a', '#fdf3ab', '#ffffcc']
mom0=mom0#[::-1]
# for velocity maps
mom1=hex_list

def vel_map(c=mom1):
	if c == 'mom0': c = mom0
	return get_continuous_cmap(c)


