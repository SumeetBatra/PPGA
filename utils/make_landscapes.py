import os
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def make_landscapes():
    filepath = '/home/sumeet/QDPPO/experiments/debug/0/archive_snapshot.csv'
    outdir = '/home/sumeet/QDPPO/experiments/debug/0/surfaces'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    df = pd.read_csv(filepath)
    num_rows = len(df)
    for row in range(num_rows):
        data = df.iloc[row]
        cells = data.iloc[1:].to_numpy().reshape(30, 30)
        sh0, sh1 = cells.shape[0], cells.shape[1]
        x = np.linspace(0, 1, sh0)
        y = np.linspace(0, 1, sh1)

        fig = go.Figure(data=[go.Surface(z=cells, x=x, y=y)])
        fig.update_layout(title='Archive 3D Surface', autosize=False, width=800, height=600,
                          margin=dict(l=65, r=50, b=65, t=90))

        filepath = os.path.join(outdir, f'archive_surface_{row:05d}.png')
        fig.write_image(filepath)
        print(f'Finished image {row}')


def frame_args(duration):
    return {
        "frame": {"duration": duration},
        "mode": "immediate",
        "fromcurrent": True,
        "transition": {"duration": duration, "easing": "linear"}
    }


def animated_surface():
    filepath = '/home/sumeet/QDPPO/experiments/debug/0/archive_snapshots.csv'
    outdir = './videos'
    df = pd.read_csv(filepath)
    z_data = df.drop('Iteration', axis=1).to_numpy().reshape(-1, 30, 30)
    sh0, sh1 = z_data.shape[1], z_data.shape[2]
    x = np.linspace(0, 1, sh0)
    y = np.linspace(0, 1, sh1)

    fps = 30
    duration_ms = 1000 * (1 / fps)

    fig = go.Figure(
        data=[go.Surface(z=z_data[0])],
        layout=go.Layout(updatemenus=[dict(type='buttons', buttons=[dict(label="Play", method='animate', args=
        [None, {"frame": {"duration": duration_ms, 'redraw': True}}])])]),
        frames=[go.Frame(data=[go.Surface(z=k)], name=str(i)) for i, k in enumerate(z_data)]
    )

    # fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="tomato", project_z=True), colorscale='portland')
    fig.update_layout(title='Archive 3D Surface', autosize=False, width=1920, height=1080,
                      margin=dict(l=65, r=50, b=65, t=90))
    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]

    fig.update_layout(sliders=sliders)
    fig.show()


if __name__ == '__main__':
    animated_surface()
