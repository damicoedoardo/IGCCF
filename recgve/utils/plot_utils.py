import os
STYLESHEET_PATH = os.environ["HOME"] + "/GitProjects/RecGVE/matplotlib_styles/"

def setup_plot(w_pts=506, fig_ratio=0.69, font_size=None, dpi=None, style_sheet="prova"):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import os

    # Convert pt to inches
    inches_per_pt = 1.0 / 72.27
    # first set the mpl defaults, then␣,!load our style
    mpl.rcdefaults()
    plt.style.use(
        STYLESHEET_PATH + style_sheet + ".mplstyle"
        #os.environ["HOME"] + "/.matplotlib/stylelib/paperstyle_notex.mplstyle"
    )
    plt.rc('font', family='sans-serif')
    # Sometime need to quickly adjust font size! so include it as an option...
    if font_size is not None:
        mpl.rcParams.update({"font.size": font_size})
    # convert pts to matplotlib! dimensions
    w = w_pts * inches_per_pt
    h = w * fig_ratio
    # dpi only matters for png
    dpi = 120 if dpi is None else dpi
    mpl.rcParams.update({"figure.figsize": (w, h)})
    mpl.rcParams.update({"figure.dpi": dpi})
    return

