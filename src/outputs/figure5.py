def auto_fit_text(text, width, height, fig=None, ax=None):
    '''Auto-decrease the fontsize of a text object.

    Args:
        text (matplotlib.text.Text)
        width (float): allowed width in data coordinates
        height (float): allowed height in data coordinates
    '''
    fig = fig or plt.gcf()
    ax = ax or plt.gca()

    # get text bounding box in figure coordinates
    renderer = fig.canvas.get_renderer()
    bbox_text = text.get_window_extent(renderer=renderer)

    # transform bounding box to data coordinates
    bbox_text = Bbox(ax.transData.inverted().transform(bbox_text))

    # evaluate fit and recursively decrease fontsize until text fits
    fits_width = bbox_text.width < width if width else True
    fits_height = bbox_text.height < height if height else True
    if not all((fits_width, fits_height)):
        new_text = text.get_text()[:-1]
        text.set_text(new_text)
        if len(new_text) > 0:
            auto_fit_text(text, width, height, fig, ax)


def make_stacked_barchart(ax, df):
    # columns are topics/communities/, rows are taxa
    k = len(df.columns)
    v = len(df.index)
    tcd = {t: c for t, c in zip(df.index, taxon_colors[:len(df.index)])}

    fig, ax = plt.subplots() if ax is None else (ax.get_figure(), ax)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.5, k + 0.5])
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks(list(range(1, k + 1)))
    ax.set_yticklabels(df.columns)
    bar_height = 0.9
    for i, topic in enumerate(df.columns):
        x = 0.0
        y = k - i - 0.45
        for taxon, prob in df[topic].sort_values(ascending=False).items():
            ax.add_artist(Rectangle((x, y), prob, bar_height, fill=True, ec='k', fc=tcd[taxon]))
            if prob > 0.05:
                text_label = ax.text(x + prob / 2, y + bar_height / 2, taxon, ha='center', va='center')
                auto_fit_text(text_label, prob, bar_height, fig, ax)
            x += prob
    return ax