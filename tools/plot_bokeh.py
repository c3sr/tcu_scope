#! /usr/bin/env python3

import json
import sys
import pprint
import os
import re
import numpy as np
import yaml
import glob
from cycler import cycler
from bokeh.layouts import gridplot
from bokeh.plotting import figure, save, output_file
from bokeh.models import Range1d, Label
from bokeh.models import scales


pp = pprint.PrettyPrinter(indent=4)

linestyles = ['solid', 'dashed', 'dotted', 'dotdash', 'dashdot']*2
markers = ["circle", "square", "triangle", "asterisk",
           "circle_x", "square_x", "inverted_triangle", "diamond"]
colors = ['#E24A33', '#348ABD', '#988ED5', '#777777',
                     '#FBC15E', '#8EBA42', '#FFB5B8', '#7F7F7F']


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

    num_series = len(cfg["series"])

    default_file = cfg.get("input_file", "not_found")
    default_plot_type = cfg.get("plot_type", "line")
    default_x_scale = eval(str(cfg.get("xaxis", {}).get("scale", 1.0)))
    default_y_scale = eval(str(cfg.get("yaxis", {}).get("scale", 1.0)))

    default_x_field = cfg.get("xaxis", {}).get("field", "real_time")
    default_y_field = cfg.get("yaxis", {}).get("field", "real_time")

    y_axis_label = ""
    y_scale = "linear"
    y_range = None
    if "yaxis" in cfg:
        axis_cfg = cfg["yaxis"]
        if axis_cfg and "lim" in axis_cfg:
            if len(axis_cfg["lim"]) == 2:
                y_range = axis_cfg["lim"]
        if axis_cfg and "scaling_function" in axis_cfg:
            y_scale = axis_cfg["scaling_function"]
            if y_scale == "log10":
                y_scale = "log"
        if axis_cfg and "label" in axis_cfg:
            y_axis_label = axis_cfg["label"]

    x_axis_label = ""
    x_scale = "linear"
    x_range = None
    x_scale_base = None
    if "xaxis" in cfg:
        axis_cfg = cfg["xaxis"]
        if axis_cfg and "lim" in axis_cfg:
            if len(axis_cfg["lim"]) == 2:
                x_range = axis_cfg["lim"]
        if axis_cfg and "scaling_function" in axis_cfg:
            x_scale = axis_cfg["scaling_function"]
            if x_scale == "log10":
                x_scale = "log"
            else:
                x_scale = "log"
                x_scale_base = "log2"
        if axis_cfg and "label" in axis_cfg:
            x_axis_label = axis_cfg["label"]
    title = ""
    if "title" in cfg:
        title = cfg["title"]

    fig = figure(title=title,
                 x_axis_label=x_axis_label,
                 y_axis_label=y_axis_label,
                 x_axis_type=x_scale,
                 y_axis_type=y_scale,
                 x_range=x_range,
                 y_range=y_range,
                 plot_width=808,
                 plot_height=int(500/2.0),
                 toolbar_location='above',
                 sizing_mode='scale_width'
                 )
    if x_scale_base == "log2":
        fig.xaxis[0].ticker.base = 2
        fig.xaxis[0].formatter.ticker = fig.xaxis[0].ticker

    # horizLabel = LatexLabel(text=x_axis_label,
    #                         x=500, y=0,
    #                         x_units='screen', y_units='screen',
    #                         render_mode='css', text_font_size='6pt',
    #                         angle=0)
    # fig.add_layout(horizLabel, "below")

    # vertLabel = LatexLabel(text=y_axis_label,
    #                        x=50, y=500,
    #                        x_units='screen', y_units='screen',
    #                        render_mode='css', text_font_size='6pt',
    #                        angle=90, angle_units="deg")
    # fig.add_layout(vertLabel, "left")

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

        # pp.pprint(y)
        fig.line(x, y,
                 legend=label,
                 line_cap="round",
                 line_width=3,
                 line_color=(colors[c % len(colors)]),
                 line_dash=(linestyles[c % len(linestyles)]),
                 )
        fig.scatter(x, y, marker=(markers[c % len(markers)]), size=7,
                    line_color=(colors[c % len(colors)]),
                    fill_color=(colors[c % len(colors)]), alpha=1)

    default_legend_ncols = max(min(int(num_series / 4), 4), 1)

    fig.legend.location = "top_left"

    if fig is not None:
        # Save plot
        xprint("saving to", output_path)
        # fig.show()
        for output_path in output_paths:
            if output_path.endswith(".html"):
                output_file(output_path)
                save(fig)


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
