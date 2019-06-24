#! /usr/bin/env python3

import json
import sys
import pprint
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import yaml
import glob
from cycler import cycler
from multiprocessing import Pool
import mpld3
from mpld3 import plugins


pp = pprint.PrettyPrinter(indent=4)

plot_theme = "seaborn-colorblind"
cycles = cycler('color', ['#284595', '#465362', '#94C595', '#FD6A02',
                          '#000000', '#D51745', '#FFB5B8', '#7F7F7F',
                          '#7F7F7F', '#FFB5B8', '#8EBA42', '#FBC15E',
                          '#777777', '#988ED5', '#348ABD', '#E24A33'])
cycles += cycler('linestyle', ['-', '--', ':', '-.']*4)
# cycles += cycler('marker', ['>', '^', '+', '*', '.', ',', 'D', 'o']*2)


plot_scale_factor = 2
plot_style = {
    "xtick.labelsize": (plot_scale_factor/1.5)*20,
    "ytick.labelsize": (plot_scale_factor/1.5)*20,
    "font.size": (plot_scale_factor/1.5)*20,
    "figure.autolayout": True,
    "figure.figsize": (2*9.7, 6.0),  # golden ratio proportion
    "axes.titlesize": (plot_scale_factor/1.5)*20,
    "axes.labelsize": (plot_scale_factor/1.5)*18,
    "lines.linewidth": plot_scale_factor*5,
    "lines.markersize": plot_scale_factor,
    "legend.fontsize": (plot_scale_factor/1.5)*22,
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "legend.handlelength": 5.0,
    'text.usetex': True,
    'text.latex.unicode': True,
    'errorbar.capsize': 3,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'ytick.color': '#000000',
    'xtick.color': '#000000',
    'text.latex.preamble': r'\usepackage{{amsmath}},\usepackage{{amsfonts}},\usepackage{{amssymb}},\usepackage{{bm}}',
    'axes.prop_cycle': cycles

}

plt.Locator.MAXTICKS = 10000


def xprint(*args):
    return


def iplot(yaml_path, output_path=None):

    root_dir = os.path.dirname(os.path.abspath(yaml_path))
    data_output_path = output_path

    # load the config
    with open(yaml_path, "rb") as f:
        cfg = yaml.load(f)

    if output_path is None and cfg.get("output_file", None) is not None:
        output_path = os.path.join(root_dir, cfg.get("output_file"))
        data_output_path = output_path + ".json"
        if not output_path.endswith(".pdf") and not output_path.endswith(".png"):
            base_output_path = output_path
            output_path = []
            for ext in cfg.get("output_format", ["pdf"]):
                ext = ext.rstrip(".")
                output_path.append(base_output_path + "." + ext)

    output_paths = [output_path] if type(
        output_path) == list() else output_path

    plt.style.use(plot_theme)
    plt.style.use(plot_style)

    fig = plt.figure()
    fig.autofmt_xdate()

    data = []
    ax = fig.add_subplot(1, 1, 1, aspect='auto')
    bar_width = cfg.get("bar_width", 0.8)
    num_series = len(cfg["series"])

    default_file = cfg.get("input_file", "not_found")
    default_plot_type = cfg.get("plot_type", "line")
    default_x_scale = eval(str(cfg.get("xaxis", {}).get("scale", 1.0)))
    default_y_scale = eval(str(cfg.get("yaxis", {}).get("scale", 1.0)))

    default_x_field = cfg.get("xaxis", {}).get("field", "real_time")
    default_y_field = cfg.get("yaxis", {}).get("field", "real_time")

    for c, s in enumerate(cfg["series"]):
        file_path = s.get("input_file", default_file)
        label = s["label"]
        xprint(label)
        regex = s.get("regex", ".*")
        if not regex.startswith("^"):
            regex = "^" + regex
        xprint("Using regex:", regex)
        yscale = eval(str(s.get("yscale", default_y_scale)))
        xscale = eval(str(s.get("xscale", default_x_scale)))
        if not os.path.isabs(file_path):
            file_path = os.path.join(root_dir, file_path)
        xprint("reading", file_path)
        with open(file_path, "rb") as f:
            j = json.loads(f.read().decode("utf-8"))

        pattern = re.compile(regex)
        matches = [
            b for b in j["benchmarks"] if (pattern == None or pattern.search(b["name"])) and b.get("error_occurred", False) == False
        ]
        times = matches

        if len(times) == 0:
            print("unable to find data for " + label + " using " + regex)
            continue

        for idx, time in enumerate(times):
            try:
                extra_keys = time["name"].split("/")[1:]
                for keyval in extra_keys:
                    if keyval == "manual_time":
                        continue
                    es = keyval.split(":")
                    if es[0] not in time:
                        time[es[0]] = eval(es[1])
            except Exception as err:
                None

        xfield = s.get("xfield", default_x_field)
        yfield = s.get("yfield", default_y_field)

        if not xfield in times[0]:
            print("unable to find the xfield " + xfield + " in " + label)
            continue
        if not yfield in times[0]:
            print("unable to find the yfield " + yfield + " in " + label)
            continue

        x = np.array(
            [None if ("error_message" in b) or (yfield not in b) or (xfield not in b) or (float(b[xfield]) == 0)or (float(b[yfield]) == 0) else float(b[xfield]) for b in times]).astype(np.double)
        xmask = np.isfinite(x)

        y = np.array(
            [None if ("error_message" in b) or (yfield not in b) or (xfield not in b) or (float(b[xfield]) == 0) or (float(b[yfield]) == 0) else float(b[yfield]) for b in times]).astype(np.double)

        ymask = np.isfinite(y)

        # Rescale
        x *= xscale
        y *= yscale

        x, y = zip(*sorted(zip(x[xmask].tolist(), y[ymask].tolist())))

        # for m, n in zip(x, y):
        #     pp.pprint((m, n))

        # pp.pprint(times)
        # if c == 0:
        #     ax.set_xlim(min(x), max(x))

        # pp.pprint(y)
        plot_type = s.get("plot_type", default_plot_type)
        if plot_type == "bar":
            ax.bar(x + bar_width * c, y, width=bar_width,
                   label=label, align="center")
            ax.set_xticklabels((x + c * bar_width).round(2))
            ax.set_xticks(x + 1.5 * bar_width, minor=False)
        else:
            ax.plot(x, y,  label=label)

        data.append({'label': label, 'series': {
                    'x': x, 'y': y}})

    if "yaxis" in cfg:
        axis_cfg = cfg["yaxis"]
        if axis_cfg and "lim" in axis_cfg:
            lim = axis_cfg["lim"]
            xprint("setting ylim", lim)
            if len(lim) == 0:
                None
            elif len(lim) == 1:
                ax.set_ylim(bottom=lim[0])
            elif len(lim) == 2:
                ax.set_ylim(lim)
            else:
                print("expecting 1 or 2 elements for ylim")
        if axis_cfg and "scaling_function" in axis_cfg:
            scale = axis_cfg["scaling_function"]
            xprint("setting xscale", scale)
            if scale == "log10":
                ax.set_yscale("log", basey=10)
            else:
                ax.set_yscale(scale, basey=2)
        if axis_cfg and "label" in axis_cfg:
            label = axis_cfg["label"]
            xprint("setting ylabel", label)
            ax.set_ylabel(label)

    if "xaxis" in cfg:
        axis_cfg = cfg["xaxis"]
        if axis_cfg and "lim" in axis_cfg:
            lim = axis_cfg["lim"]
            xprint("setting xlim", lim)
            if len(lim) == 0:
                None
            elif len(lim) == 1:
                ax.set_xlim(left=lim[0])
            elif len(lim) == 2:
                ax.set_xlim(lim)
            else:
                print("expecting 1 or 2 elements for ylim")
        if axis_cfg and "scaling_function" in axis_cfg:
            scale = axis_cfg["scaling_function"]
            xprint("setting xscale", scale)
            if scale == "log10":
                ax.set_xscale("log", basex=10)
            else:
                ax.set_xscale(scale, basex=2)
        if axis_cfg and "label" in axis_cfg:
            label = axis_cfg["label"]
            xprint("setting xlabel", label)
            ax.set_xlabel(label)

    # if "title" in cfg:
    #     title = cfg["title"]
    #     xprint("setting title", title)
    #     ax.set_title(title)

    # default_legend_location = 'upper left'
    default_legend_location = 'best'

    default_legend_ncols = max(min(int(num_series / 4), 4), 1)

    ax.legend(loc=cfg.get('legend_location', default_legend_location),
              ncol=cfg.get('legend_columns', default_legend_ncols),
              fontsize='small')

    fig.set_tight_layout(True)

    plt.gcf().tight_layout()

    with open(data_output_path, 'w') as fp:
        json.dump(data, fp)

    if fig is not None:
        # Save plot
        xprint("saving to", output_path)
        # fig.show()
        for output_path in output_paths:
            if output_path.endswith(".html"):
                handles, labels = ax.get_legend_handles_labels()  # return lines and labels
                interactive_legend = plugins.InteractiveLegendPlugin(zip(handles,
                                                                         ax.collections),
                                                                     labels,
                                                                     alpha_unsel=1.5,
                                                                     alpha_over=1.5,
                                                                     start_visible=True)
                plugins.connect(fig, interactive_legend)
                d = mpld3.fig_to_dict(fig)
                with open(output_path.rstrip(".html") + ".dict.json", 'w') as fp:
                    json.dump(d, fp)
                # html = mpld3.fig_to_html(fig, template_type="general")
                # with open(output_path, 'w') as fp:
                #     fp.write(html)
            else:
                fig.savefig(output_path, clip_on=False,
                            transparent=False)
        plt.close(fig)


def plot(yaml_path, output_path=None):
    try:
        iplot(yaml_path, output_path=output_path)
    except Exception as err:
        print("unable to plot " + yaml_path + " because " + str(err))


def main():
    if len(sys.argv) == 2:
        yaml_path = sys.argv[1]
        if yaml_path != "all":
            return plot(yaml_path)

    if len(sys.argv) == 1 or len(sys.argv) == 2:
        this_directory = os.path.dirname(os.path.realpath(__file__))
        for yaml_path in glob.glob(os.path.join(this_directory, "..", "docs", "figures", "spec", "*yml")):
            print("plotting " + yaml_path + "  ...")
            plot(yaml_path)
        return

    if len(sys.argv) == 3:
        output_path = sys.argv[1]
        yaml_path = sys.argv[2]
        return plot(yaml_path, output_path=output_path)

    print("invalid number of arguments")
    sys.exit(1)


if __name__ == "__main__":
    main()
