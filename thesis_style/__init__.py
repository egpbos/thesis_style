# -*- coding: utf-8 -*-


from .__version__ import __version__

__author__ = "E. G. Patrick Bos"
__email__ = 'egpbos@gmail.com'


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import math
import seaborn.apionly as sns

import os
this_path = os.path.abspath(os.path.dirname(__file__))

thesis_mplstyle = os.path.join(this_path, 'thesis.mplstyle')


golden_ratio = (np.sqrt(5.0)-1.0)/2.0

inch_per_pt = 1./72.

# AxesGrid parameters that cannot be set in rcParams
grid_pad = 0.05
cbar_size = grid_pad / golden_ratio

# dashed lines
grey = (0.85,0.85,0.85)
greyhex = mpl.colors.rgb2hex(grey)
black = (0.15,0.15,0.15)
blackhex = mpl.colors.rgb2hex(black)

dashed = {'lw':0.87, 'ls': (0, (2.0, 2.0))}
dashed_grey = {'lw':0.87, 'ls': (0, (2.0, 2.0)), 'color': grey}
dashed_black = {'lw':0.87, 'ls': (0, (2.0, 2.0)), 'color': black}

# cmaps
# seaborn default colors, named for convenience
with mpl.style.context(['seaborn-paper', 'seaborn-dark', 'seaborn-deep',
                        thesis_mplstyle]):
    sns_colors = sns.color_palette()
    sns_chex = {'b': mpl.colors.rgb2hex(sns_colors[0]),
                'g': mpl.colors.rgb2hex(sns_colors[1]),
                'r': mpl.colors.rgb2hex(sns_colors[2]),
                'p': mpl.colors.rgb2hex(sns_colors[3]),
                'y': mpl.colors.rgb2hex(sns_colors[4]),
                'c': mpl.colors.rgb2hex(sns_colors[5])}
# grays
gray_cmap = mpl.colors.ListedColormap(sns.color_palette("gray", 256))
gray_r_cmap = mpl.colors.ListedColormap(sns.color_palette("gray_r", 256))


def axgrid(rowcol, mode, fig=None, subfig=111, cbar_location="right",
           add_all=True, cbar_pad=grid_pad, cbar_size=cbar_size):
    if fig is None:
        fig = plt.figure()

    grid = AxesGrid(fig, subfig,
                    nrows_ncols=rowcol,
                    axes_pad=grid_pad,
                    label_mode="L",
                    share_all=False,
                    cbar_location=cbar_location,
                    cbar_mode=mode,
                    cbar_size=cbar_size,
                    cbar_pad=cbar_pad,
                    add_all=add_all)

    return fig, grid

def axgrid_rows(rowcol, **kwargs):
    return axgrid(rowcol, "edge", **kwargs)


def axgrid_same(rowcol, **kwargs):
    return axgrid(rowcol, "single", **kwargs)


def axgrid_cbar_text_location(grid, cbar_location=None):
    if cbar_location is None:
        cbar_location = grid._colorbar_location
    for ix in range(len(grid.cbar_axes)):
        if cbar_location in ["right", "left"]:
            grid.cbar_axes[ix].yaxis.set_ticks_position(cbar_location)
            grid.cbar_axes[ix].yaxis.set_label_position(cbar_location)
        else:
            grid.cbar_axes[ix].xaxis.set_ticks_position(cbar_location)
            grid.cbar_axes[ix].xaxis.set_label_position(cbar_location)


# axgrid_rows = lambda rowcol, fig=None: axgrid(rowcol, "edge", fig=fig)
# axgrid_same = lambda rowcol, fig=None: axgrid(rowcol, "single", fig=fig)


def hide_overlapping_axgrid_xlabels(grid):
    rows, cols = grid.get_geometry()

    for col in range(1, cols):
        axcol = grid.axes_column[col]
        for row in range(rows):
            xlabels = axcol[row].get_xticklabels()
            if xlabels:
                left_xtick = xlabels[0]
                plt.setp(left_xtick, visible=False)


def hide_overlapping_axgrid_ylabels(grid):
    rows, cols = grid.get_geometry()

    for row in range(rows-1):
        axrow = grid.axes_row[row]
        for col in range(cols):
            ylabels = axrow[col].get_yticklabels()
            if ylabels:
                bottom_ytick = ylabels[0]
                plt.setp(bottom_ytick, visible=False)

    
def hide_overlapping_axgrid_cblabels(grid):
    rows, cols = grid.get_geometry()

    if grid._colorbar_mode == "edge":
        if grid._colorbar_location in ("left", "right"):
            for row in range(rows-1):
                bottom_cbtick = grid.cbar_axes[row].get_yticklabels()[0]
                plt.setp(bottom_cbtick, visible=False)
        else:
            for col in range(1, cols):
                left_cbtick = grid.cbar_axes[col].get_xticklabels()[0]
                plt.setp(left_cbtick, visible=False)
    elif grid._colorbar_mode == "each":
        if grid._colorbar_location in ("left", "right"):
            for col in range(cols):
                for row in range(rows-1):
                    bottom_cbtick = grid.axes_row[row][col].cax.get_yticklabels()[0]
                    plt.setp(bottom_cbtick, visible=False)
        else:
            for row in range(rows):
                for col in range(1, cols):
                    left_cbtick = grid.axes_row[row][col].cax.get_xticklabels()[0]
                    plt.setp(left_cbtick, visible=False)


def hide_overlapping_axgrid_labels(grid, cb=True):
    """Use on AxesGrid/ImageGrid after plotting everything."""
    hide_overlapping_axgrid_xlabels(grid)
    hide_overlapping_axgrid_ylabels(grid)
    if cb:
        hide_overlapping_axgrid_cblabels(grid)


def fit_height(fig, x_offset=0, y_offset=0, x_stretch=0, y_stretch=0, h_stretch=0):
    # reduce figure height to reduce remaining white space
    tight_bbox_raw = fig.get_tightbbox(fig.canvas.get_renderer())
    height_tight = tight_bbox_raw.bounds[3]
    fig.set_size_inches(fig.get_size_inches()[0], height_tight + h_stretch)

    fig.subplots_adjust(top=fig.subplotpars.top + y_offset + y_stretch,
                        bottom=fig.subplotpars.bottom + y_offset - y_stretch,
                        right=fig.subplotpars.right + x_offset + x_stretch,
                        left=fig.subplotpars.left + x_offset - x_stretch)


def pretty_axgrid(fig, grid, x_offset=0, y_offset=0, x_stretch=0, y_stretch=0,
                  hide_labels=True, hide_cb_labels=True, h_stretch=0):
    """Run before showing/saving the figure with an AxesGrid/ImageGrid."""

    # hide overlapping labels
    if hide_labels:
        hide_overlapping_axgrid_labels(grid, cb=hide_cb_labels)

    # fix colorbar text location for grids where it is not right
    axgrid_cbar_text_location(grid)

    # reduce white space as much as possible without changing figure size
    fig.tight_layout(pad=0.1)
    fit_height(fig, x_offset=x_offset, y_offset=y_offset, x_stretch=x_stretch,
               y_stretch=y_stretch, h_stretch=h_stretch)


def reformat_axes(ax):
    rcp = plt.rcParams
    plt.setp(ax, axis_bgcolor=rcp['axes.facecolor'])
    spine_flags = {k[12:]: v for k, v in rcp.iteritems() if 'axes.spines.' in k}
    for s_ix, s_f in spine_flags.iteritems():
        plt.setp(ax.spines[s_ix], visible=s_f, edgecolor=rcp['axes.edgecolor'])


def use_dark_text(bgcolor):
    """
    Calculates whether to use a dark (True) or light (False) text color on the
    background color `bgcolor` (rgba array; a(lpha) is not used though!).
    Based on https://24ways.org/2010/calculating-color-contrast
    """
    r, g, b, a = mpl.colors.colorConverter.to_rgba(bgcolor)
    r *= 256
    g *= 256
    b *= 256
    yiq = ((r*299)+(g*587)+(b*114))/1000;
    if yiq >= 128:
        return True
    else:
        return False


def contrasting_text_color(bgcolor, dark=black + (1.,), light=grey + (1.,)):
    if use_dark_text(bgcolor):
        return dark
    else:
        return light


def footnotesize(fontsize):
    if fontsize == 10:
        return 8
    elif fontsize in [10.95, 11]:
        return 9
    elif fontsize == 12:
        return 10
    else:
        raise Exception("footnotesize: fontsize %s not supported!" % str(fontsize))


def footnotesize_inch(fontsize):
    return footnotesize(fontsize) * inch_per_pt


def compactify_colorbar(cb, cax):
    """
    N.B.: this function assumes a colorbar of size footnotesize! Resize your
    colorbar or create it with axgrid by specifying the cbar_size (in inches).
    """
    color_min = contrasting_text_color(cb.get_cmap().colors[0])
    color_max = contrasting_text_color(cb.get_cmap().colors[-1])

    font_size = footnotesize(plt.rcParams['font.size'])

    if cax.orientation in ['top', 'bottom']:
        cb_vmin_lab = cax.get_xticklabels()[0]
        cb_vmax_lab = cax.get_xticklabels()[-1]

        cax.get_xaxis().set_ticks([])
        lab_left  = cax.text(0, .5, cb_vmin_lab.get_text(), color=color_min,
                             ha='left', va='center', zorder=1, size=font_size)
        lab_right = cax.text(1, .5, cb_vmax_lab.get_text(), color=color_max,
                             ha='right', va='center', zorder=1, size=font_size)
    else:
        cb_vmin_lab = cax.get_yticklabels()[0]
        cb_vmax_lab = cax.get_yticklabels()[-1]

        cax.get_yaxis().set_ticks([])
        lab_bottom  = cax.text(.5, 0, cb_vmin_lab.get_text(), color=color_min,
                               ha='center', va='bottom', zorder=1, size=font_size)
        lab_top = cax.text(.5, 1, cb_vmax_lab.get_text(), color=color_max,
                           ha='center', va='top', zorder=1, size=font_size)


def compactify_colorbars(grid):
    for ax in grid:
        cb = ax.get_images()[0].colorbar
        compactify_colorbar(cb, ax.cax)


def bold_string(s):
    s = str(s)
    s = r'\textbf{' + s + r'}'
    return s


def border_string(s, bold=False):
    # http://stackoverflow.com/a/12958839/1199693
    if not '\definecolor{seaborn-white}' in ''.join(plt.rcParams['pgf.preamble']):
        raise Exception("The color seaborn-white is not defined in the pgf.preamble!")
    if not '\definecolor{seaborn-black}' in ''.join(plt.rcParams['pgf.preamble']):
        raise Exception("The color seaborn-black is not defined in the pgf.preamble!")

    s = str(s)
    if bold:
        s = bold_string(s)
    s = r'\contour{seaborn-black}{\textcolor{seaborn-white}{' + s + r'}}'

    return s


def inside_title(ax, title, loc=(0.5, 0.9), horizontalalignment='center',
                 bold=True, border=True, **kwargs):
    # http://stackoverflow.com/a/12958839/1199693
    if not '\definecolor{seaborn-white}' in ''.join(plt.rcParams['pgf.preamble']):
        raise Exception("The color seaborn-white is not defined in the pgf.preamble!")

    if bold:
        title = bold_string(title)
    if border:
        title = border_string(title, bold=False)
    else:
        title = r'{\textcolor{seaborn-white}{' + title + r'}}'
    t = ax.text(loc[0], loc[1], title, horizontalalignment=horizontalalignment,
                transform=ax.transAxes, **kwargs)
    return t


# def inside_title_grey(ax, title, **kwargs):
#     # http://stackoverflow.com/a/12958839/1199693
#     t = inside_title(ax, title, color=grey, **kwargs)
#     return t


class InvLogLocator(mpl.ticker.LogLocator):
    """
    Determine the tick locations for inverse log axes
    """

    def __init__(self, base=10.0, inv_base=10.0, inv_factor=1.,
                 subs=[1.0], numdecs=4, numticks=15):
        """
        place ticks on the location= base**i*subs[j]
        Set subs to None for autosub (for minor locator).
        """
        self.base(base)
        self.inv_base(inv_base)
        self.inv_factor(inv_factor)
        self.subs(subs)
        self.numticks = numticks
        self.numdecs = numdecs

    def set_params(self, base=None, inv_base=None, inv_factor=None, subs=None,
                   numdecs=None, numticks=None):
        """Set parameters within this locator."""
        if base is not None:
            self.base = base
        if inv_base is not None:
            self.inv_base = inv_base
        if inv_factor is not None:
            self.inv_factor = inv_factor
        if subs is not None:
            self.subs = subs
        if numdecs is not None:
            self.numdecs = numdecs
        if numticks is not None:
            self.numticks = numticks

    def inv_base(self, inv_base):
        """
        set the base of the log scaling (major tick every base**i, i integer)
        for the inverse of the data
        """
        self._inv_base = inv_base + 0.0

    def inv_factor(self, inv_factor):
        """
        After inverting the data, it can be multiplied by this factor. This way,
        any inverse logarithmic quantity can be computed from a logarithmic
        input quantity.
        """
        self._inv_factor = inv_factor + 0.0

    def __call__(self):
        'Return the locations of the ticks'
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        b = self._base
        ib = self._inv_base
        # max and min swapped
        ivmax = self._inv_factor / vmin
        ivmin = self._inv_factor / vmax
        # dummy axis has no axes attribute
        if hasattr(self.axis, 'axes') and self.axis.axes.name == 'polar':
            ivmax = math.ceil(math.log(ivmax) / math.log(ib))
            decades = np.arange(ivmax - self.numdecs, ivmax)
            i_ticklocs = ib ** decades

            ticklocs = self._inv_factor / i_ticklocs

            return ticklocs

        if ivmax <= 0.0:
            if self.axis is not None:
                ivmax = self._inv_factor / self.axis.get_minpos()

            if ivmax <= 0.0 or not np.isfinite(ivmax):
                raise ValueError(
                    "Data has no positive values, and therefore can not be "
                    "log-scaled.")

        ivmin = math.log(ivmin) / math.log(ib)
        ivmax = math.log(ivmax) / math.log(ib)

        if ivmax < ivmin:
            ivmin, ivmax = ivmax, ivmin

        numdec = math.floor(ivmax) - math.ceil(ivmin)

        if self._subs is None:  # autosub
            if numdec > 10:
                subs = np.array([1.0])
            elif numdec > 6:
                subs = np.arange(2.0, ib, 2.0)
            else:
                subs = np.arange(2.0, ib)
        else:
            subs = self._subs

        stride = 1
        while numdec / stride + 1 > self.numticks:
            stride += 1

        decades = np.arange(math.floor(ivmin) - stride,
                            math.ceil(ivmax) + 2 * stride, stride)
        if hasattr(self, '_transform'):
            i_ticklocs = self._transform.inverted().transform(decades)
            if len(subs) > 1 or (len(subs == 1) and subs[0] != 1.0):
                i_ticklocs = np.ravel(np.outer(subs, ticklocs))
        else:
            if len(subs) > 1 or (len(subs == 1) and subs[0] != 1.0):
                i_ticklocs = []
                for decadeStart in ib ** decades:
                    i_ticklocs.extend(subs * decadeStart)
            else:
                i_ticklocs = ib ** decades

        ticklocs = self._inv_factor / np.asarray(i_ticklocs)

        return self.raise_if_exceeds(np.asarray(ticklocs))

    # def view_limits(self, vmin, vmax):
    #     'Try to choose the view limits intelligently'
    #     b = self._base

    #     if vmax < vmin:
    #         vmin, vmax = vmax, vmin

    #     if self.axis.axes.name == 'polar':
    #         vmax = math.ceil(math.log(vmax) / math.log(b))
    #         vmin = b ** (vmax - self.numdecs)
    #         return vmin, vmax

    #     minpos = self.axis.get_minpos()

    #     if minpos <= 0 or not np.isfinite(minpos):
    #         raise ValueError(
    #             "Data has no positive values, and therefore can not be "
    #             "log-scaled.")

    #     if vmin <= minpos:
    #         vmin = minpos

    #     if rcParams['axes.autolimit_mode'] == 'round_numbers':
    #         if not is_decade(vmin, self._base):
    #             vmin = decade_down(vmin, self._base)
    #         if not is_decade(vmax, self._base):
    #             vmax = decade_up(vmax, self._base)

    #         if vmin == vmax:
    #             vmin = decade_down(vmin, self._base)
    #             vmax = decade_up(vmax, self._base)

    #     result = mtransforms.nonsingular(vmin, vmax)
    #     return result


class InvLogFormatter(mpl.ticker.LogFormatter):
    """
    Format values for inverted log axis;
    """
    def __init__(self, base=10.0, inv_base=10.0, inv_factor=1.,
                 labelOnlyBase=True):
        """
        *base* is used to locate the decade tick,
        which will be the only one to be labeled if *labelOnlyBase*
        is ``False``
        """
        self.base(base)
        self.inv_base(inv_base)
        self.inv_factor(inv_factor)
        self.labelOnlyBase = labelOnlyBase

    def inv_base(self, inv_base):
        """
        set the base of the log scaling (major tick every base**i, i integer)
        for the inverse of the data - warning: should always match the
        base used for :class:`InvLogLocator`
        """
        self._inv_base = inv_base + 0.0

    def inv_factor(self, inv_factor):
        """
        After inverting the data, it can be multiplied by this factor. This way,
        any inverse logarithmic quantity can be computed from a logarithmic
        input quantity - warning: should always match the base used for
        :class:`InvLogLocator`
        """
        self._inv_factor = inv_factor + 0.0

    def __call__(self, x, pos=None):
        """Return the format for tick val *x* at position *pos*"""
        vmin, vmax = self.axis.get_view_interval()
        # max and min swapped
        ivmax = self._inv_factor / vmin
        ivmin = self._inv_factor / vmax
        d = abs(ivmax - ivmin)
        ib = self._inv_base
        ix = self._inv_factor / x

        if ix == 0.0:
            return '0'
        sign = np.sign(ix)

        # only label the decades
        fx = math.log(abs(ix)) / math.log(ib)
        isDecade = mpl.ticker.is_close_to_int(fx)
        if not isDecade and self.labelOnlyBase:
            s = ''
        elif ix > 10000:
            s = '%1.0e' % ix
        elif ix < 1:
            s = '%1.0e' % ix
        else:
            s = self.pprint_val(ix, d)
        if sign == -1:
            s = '-%s' % s

        return self.fix_minus(s)

    # def format_data(self, value):
    #     print "FORMAT DATAAAA"
    #     b = self.labelOnlyBase
    #     self.labelOnlyBase = False
    #     value = mpl.cbook.strip_math(self.__call__(value))
    #     self.labelOnlyBase = b
    #     return value

    # def format_data_short(self, value):
    #     print "FORMAT DATAAAA short"
    #     'return a short formatted string representation of a number'
    #     return '%-12g' % value

    # def pprint_val(self, x, d):
    #     print "pprint valleeee"


def add_ticks(axis, ticklocs, labels=None, which='major', keep_limits=True):
    """
    `axis` is an x or y axis, not an `Axes`! E.g. `ax.xaxis`.
    Note that this can undo set_xlim or set_ylim! When there are ticks outside
    of these lim ranges, they will be readded here, which apparently triggers a
    resetting of the viewing limits. If keep_limits is set to True, this will
    be prevented.
    """
    if which == 'major':
        minor = False
    elif which == 'minor':
        minor = True

    if keep_limits:
        # list used to make new copy, otherwise values will be changed!
        xlim = tuple(axis.axes.get_xlim())
        ylim = tuple(axis.axes.get_ylim())
        # cannot use axis.get_view_limits(), because we need both x and y, since
        # the set_view_interval function doesn't work properly, since the view
        # interval is an Axes property, not axis, and Axes needs both x and y.

    current_ticklocs = tuple(axis.get_ticklocs(minor=minor))
    add_ticklocs = [t for t in ticklocs if t not in current_ticklocs]
    axis.set_ticks(current_ticklocs + tuple(add_ticklocs), minor=minor)

    if keep_limits:
        axis.axes.set_xlim(xlim)
        axis.axes.set_ylim(ylim)

    if labels is not None:
        current_labels = tuple(axis.get_ticklabels(which=which))
        add_labels = [labels[ix] for ix, t in enumerate(ticklocs)
                      if t not in current_ticklocs]
        axis.set_ticklabels(current_labels + tuple(add_labels), minor=minor)


def add_scale_axis(ax, logx=False, k_ticks=[], scale_ticks=[], ticks_xlim=True,
                   label_fct=lambda k: 2*np.pi/k, label_fmt="%.1f",
                   label_inv_fct=lambda scale: 2*np.pi/scale,
                   minor_ticks=True, minor_tick_labels=True):
    # tuple used to make new copy, otherwise values will be changed!
    xlim = tuple(ax.get_xlim())

    ax2 = ax.twiny()
    ax2.spines['top'].set_visible(True)
    ax2.spines['right'].set_visible(True)

    if logx:
        ax2.set_xscale('log')
        ax2.xaxis.set_major_locator(InvLogLocator(inv_factor=2*np.pi))
        ax2.xaxis.set_major_formatter(InvLogFormatter(inv_factor=2*np.pi))

        ax2.xaxis.set_minor_locator(InvLogLocator(inv_factor=2*np.pi,
                                                  subs=[2,3,4,5,6,7,8,9]))
        ax2.xaxis.set_minor_formatter(InvLogFormatter(inv_factor=2*np.pi,
                                                      labelOnlyBase=False))

        # ax.stale = True  # doet niks
        # print ax2.xaxis.get_ticklocs()
        # for l in ax2.xaxis.get_ticklabels():
        #     print l

    add_ticks(ax.xaxis, k_ticks)
    # add_ticks(ax2.xaxis,
    #           [label_inv_fct(s) for s in scale_ticks],
    #           labels=[label_fmt % s for s in scale_ticks])

    # print [label_fmt % label_fct(s) for s in xlim]
    # print tuple(ax2.xaxis.get_ticklabels())
    #
    #
    #
    #
    #
    # TODO:
    # als je add_ticks op ax2.xaxis doet vallen alle labels weg... hoe kan dat?
    # kan de InvLogLocator/Formatter niet omgaan met extra labels? Conflicteert
    # dat ofzo?
    # 
    # Of moeten het mpl.text.Text objecten worden?
    #
    #
    #
    #
    #
    #

    # if ticks_xlim:
    #     add_ticks(ax2.xaxis, xlim, labels=[label_fmt % label_fct(s) for s in xlim])

    if not minor_tick_labels:
        ax2.set_xticklabels([], minor=True)
    if not minor_ticks:
        ax2.set_xticks([], minor=True)

    ax2.set_xlim(xlim)

    # if minor_ticks:
    #     rotation = 90
    # else:
    #     rotation = 0
    rotation = 0

    # if not logx:
    #     ax2Ticks = ax2.get_xticks()
    #     ax2Ticklabels = [label_fmt % label_fct(ki) for ki in ax2Ticks]
    #     ax2.set_xticklabels(ax2Ticklabels, minor=False, rotation=rotation)

    # if minor_ticks and minor_tick_labels and not logx:
    #     ax2TicksMinor = ax2.get_xticks(minor=True)
    #     ax2TickMinorlabels = [label_fmt % label_fct(ki) for ki in ax2TicksMinor]
    #     ax2.set_xticklabels(ax2TickMinorlabels, minor=True, rotation=rotation)
    # else:
    #     ax2.set_xticklabels([], minor=True)
    ax2.set_xlabel(r'scale ($h^{-1}$ $\mathrm{Mpc}$)')
    return ax2
