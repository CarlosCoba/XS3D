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



# ---------- moment 1 cmap --------------------------------------
blues = ["#01025a", "#193476", "#376b94", "#5ea1ba", "#a3dbe6"]
reds = ["#531132", "#800d20", "#aa2c24", "#db6d52", "#f1b691"]
black = ["#423f46"]

# A nice red black and blues
hex_list = ["#01025a", "#193476", "#376b94", "#5ea1ba", "#a3dbe6",
"#423f46","#f1b691", "#db6d52", "#aa2c24", "#800d20", "#531132"]

# improved red-back-blue colors
mom1 = [
    # ── blue side (receding, v < 0) ─────────────────────────
    "#01025a", "#0d1b68", "#193476", "#28508b",
    "#376b94", "#4b86a7", "#5ea1ba", "#81bed0", "#a3dbe6",
    # ── near-zero cluster (3 colors, smooth approach to v=0) ─
    "#62737b",  # dark slate blue-gray  (v ≈ 0⁻)
    "#423f46",  # near-black charcoal   (v = 0)
    "#7c675f",  # dark warm gray-brown  (v ≈ 0⁺)
    # ── red side (approaching, v > 0) ───────────────────────
    "#f1b691", "#e69272", "#db6d52", "#c34d3b",
    "#aa2c24", "#951d22", "#800d20", "#6a0f29", "#531132",
]


# ---------- moment 0 cmap --------------------------------------

# Yellow black
mom0 = ['#f8fafc', '#9aa5b8', '#384150', '#1b202d', '#332312', '#502d1e',
'#6f372d', '#8f403b', '#b04b45', '#cd5d4c', '#de754f', '#e58b51', '#eaa053',
'#efb759', '#f5d06b', '#fae58a', '#fdf3ab', '#ffffcc','#f2f5f9']

# Black and White
mom0 =[
    "#fffdf3",  # L*≈ 99  background / no emission
    "#f2e6dd",  # L*≈ 92
    "#dacfc7",  # L*≈ 84
    "#c3b9b2",  # L*≈ 76
    "#aca39d",  # L*≈ 68
    "#968e89",  # L*≈ 60
    "#807a75",  # L*≈ 52
    "#6b6662",  # L*≈ 44
    "#57524f",  # L*≈ 35
    "#44403d",  # L*≈ 27
    "#312e2c",  # L*≈ 19
    "#1f1d1c",  # L*≈ 11
    "#0c0b0a",  # L*≈  3  peak emission
]    

# ---------- moment 2 cmap --------------------------------------
mom2=[
 # ── low dispersion (σ ≈ 0) ───────────────────────────────
    "#f3ffff",  # near-white cyan 
    "#c9ffff",  # pale cyan
    "#b0ffff",  # light cyan
    "#a4ffff",
    "#a8ffff",
    "#aafff6",  # cyan with green tint
    "#a4ffd9",  # pale mint
    "#abffaf",  # pale green
    # ── yellow-green ─────────────────────────────────────────
    "#c2ff71",
    "#e9ff25",  # vivid yellow-green
    # ── warm yellows and oranges ─────────────────────────────
    "#ffef00",  # bright yellow
    "#ffd600",
    "#ffb900",  # golden yellow
    "#ffa100",  # amber
    "#ff8800",  # orange
    "#f66b00",  # deep orange
    "#eb5200",  # orange-red
    "#df3700",
    "#d41c00",  # bright red
    "#6d0100",  # dark red
    # ── high dispersion (added) ──────────────────────────────
    "#3d0000",  # dark maroon      
    "#1a0000",  # near-black maroon   
]    

# ---------- pvd cmap --------------------------------------
pvd = ['#FDFBF6','#F6E6D6', '#EBC7A8', '#D56D4C', '#B14A33', '#7A2E25']




def vel_map(c=mom1):
	if c == 'mom0':	c = mom0
	if c == 'pvd':	c = pvd
	if c == 'mom2':	c = mom2
	return get_continuous_cmap(c)
	
	

