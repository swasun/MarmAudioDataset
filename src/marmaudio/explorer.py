# Authors: Charly Lamothe <charlylmth_AT_gmail_DOT_com>; Paul Best <paulobest25_AT_@gmail_DOT_com>

# License: BSD (3-clause)

import numpy as np
from bokeh.layouts import row
from bokeh.models import BoxSelectTool, LassoSelectTool, ColumnDataSource
from bokeh.models.tools import (LassoSelectTool, ResetTool, WheelZoomTool, PanTool, SaveTool, HoverTool)
TOOLS_FIG1 = [PanTool, LassoSelectTool, WheelZoomTool, ResetTool, SaveTool]
TOOLS_FIG2 = [PanTool, WheelZoomTool, ResetTool, SaveTool]
from bokeh.plotting import curdoc, figure
from bokeh.transform import factor_cmap
import pandas as pd
import os
from PIL import Image
import random
from datetime import datetime
import shutil
import matplotlib.pyplot as plt


save_indices = True
data_root_path = 'MarmAudioData'
audio_folder = os.path.join(data_root_path, 'Vocalizations')
spectrogram_folder = os.path.join(data_root_path, 'spectrograms')
tsv_file = f"{data_root_path}/Annotations.tsv"
encodings_file_path = 'experiment_results/encodings_AE_marmoset_logMel128_256feat_all-segmented-vocs_fixrelu_22-01-2025-15-04.npy'

# Parameters
W = 2
H = 1
num_images = 30  # Number of images
aspect_ratio = W / H
image_scale = 0.5
dw_values = [image_scale*aspect_ratio] * num_images
dh_values = [image_scale*1] * num_images
initial_positions = [(i % 10, i // 10) for i in range(num_images)]  # Grid-like initial positions
jitter_range = 0.5  # Adjust this value to control the amount of jitter

#cmap = gen_spectrogram_cmap()

encodings = np.load(encodings_file_path, allow_pickle=True).item()
idxs, umap = encodings['idx'], encodings['umap']
df = pd.read_csv(tsv_file, sep='\t')
idxs = [os.path.basename(idx) for idx in idxs]
df = df[df['file_name'].isin(idxs)].reset_index(drop=True)
df[['umap_x', 'umap_y']] = umap

fixed_x_range = (-13, 13)
fixed_y_range = (-13, 13)

df['source_filename'] = idxs

tools_fig1 = [t() for t in TOOLS_FIG1]
tools_fig2 = [t() for t in TOOLS_FIG2]

# create the scatter plot
fig1 = figure(tools=tools_fig1, width=1024, height=1024,
        toolbar_location="above", x_axis_location=None, y_axis_location=None,
        title="UMAP embedding", x_range=fixed_x_range, y_range=fixed_y_range, output_backend='webgl')
fig1.background_fill_color = "#fafafa"
fig1.select(BoxSelectTool).select_every_mousemove = False
fig1.select(LassoSelectTool).select_every_mousemove = False
data_source_points = ColumnDataSource(df)
label_types = ['Infant', 'Phee', 'Seep', 'Trill', 'Tsik', 'Twitter']
"""points_fig_renderer = fig1.scatter('umap_x', 'umap_y', color=factor_cmap('label',
    'Category10_6', label_types), source=data_source_points, line_width=0, alpha=0.5,
    size=2, legend_group="label", muted_alpha=0.2, nonselection_alpha=0.1)"""
points_fig_renderer = fig1.circle('umap_x', 'umap_y', color=factor_cmap('label',
    'Category10_6', label_types), source=data_source_points, line_width=0, alpha=0.5,
    size=0.5, legend_group="label", muted_alpha=0.2, nonselection_alpha=0.2)
fig1.add_tools(HoverTool(tooltips = [('ID', '@source_filename')], renderers=[points_fig_renderer]))

# Generate images and positions
images = []
x_positions = []
y_positions = []
for i, (x, y) in enumerate(initial_positions):
    img = np.empty((W, H), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((W, H, 4))
    for px in range(W):
        for py in range(H):
            view[px, py, 0] = 0 # Red channel
            view[px, py, 1] = 0 # Green channel
            view[px, py, 2] = 0 # Blue channel
            view[px, py, 3] = 0 # Alpha channel
    images.append(img)
    x_positions.append(x)
    y_positions.append(y)

# Combine all images into a single ColumnDataSource
data_source_images = ColumnDataSource({
    'image': images,
    'x': [0] * len(images),
    'y': [0] * len(images),
    'dw': dw_values,
    'dh': dh_values,
    'source_filename': [''] * len(images)
})

fig2 = figure(tools=tools_fig2, width=1024, height=1024, x_range=fixed_x_range, y_range=fixed_y_range,
        toolbar_location="above", x_axis_location=None, y_axis_location=None,
        title="Spectrograms")
fig2.background_fill_color = "#fafafa"
fig2.select(BoxSelectTool).select_every_mousemove = False
fig2.select(LassoSelectTool).select_every_mousemove = False
image_fig_renderer = fig2.image_rgba(image='image', x='x', y='y', dw='dw', dh='dh', source=data_source_images)
fig2.add_tools(HoverTool(tooltips = [('ID', '@source_filename')], renderers=[image_fig_renderer]))

layout = row(fig1, fig2)

curdoc().add_root(layout)
curdoc().title = "Marmaudio Explorer"

def load_image_as_rgba(file_path):
    # Open the image using Pillow
    img = Image.open(file_path).convert("RGBA")  # Ensure image is RGBA

    # Convert to numpy array
    img_np = np.array(img)  # Shape: (height, width, 4)

    # Flip the image vertically to match Bokeh's expected orientation
    img_np = np.flipud(img_np)

    # Pack the RGBA image into uint32
    img_packed = (
        (img_np[:, :, 3].astype(np.uint32) << 24) |  # Alpha
        (img_np[:, :, 2].astype(np.uint32) << 16) |  # Blue
        (img_np[:, :, 1].astype(np.uint32) << 8) |   # Green
        (img_np[:, :, 0].astype(np.uint32))          # Red
    )

    return img_packed

def convert_grayscale_to_rgba_with_cmap(img_array, cmap):
    """
    Convert a 2D grayscale image to RGBA format using a specified colormap.
    
    Args:
        img_array (numpy.ndarray): A 2D NumPy array representing a grayscale image.
        cmap_name (str): The name of the colormap to use. Defaults to 'viridis'.
    
    Returns:
        numpy.ndarray: A packed uint32 array with RGBA values.
    """
    # Ensure the grayscale image is in the range [0, 1]
    img_array_normalized = img_array / 255.0  # Normalize to [0, 1]
    
    # Apply the colormap to the normalized grayscale image
    # The result is a (height, width, 4) array in the [0, 1] range
    rgba_image = cmap(img_array_normalized)  # This is in [0, 1] RGBA format
    
    # Convert the RGBA values to the 0-255 range and convert to uint8
    rgba_image = (rgba_image[:, :, :3] * 255).astype(np.uint8)  # RGB in uint8
    alpha_channel = (rgba_image[:, :, 3] * 255).astype(np.uint8) if rgba_image.shape[2] == 4 else 255
    
    # Pack the RGBA image into uint32
    img_packed = (
        (rgba_image[:, :, 3].astype(np.uint32) << 24) |  # Alpha
        (rgba_image[:, :, 2].astype(np.uint32) << 16) |  # Blue
        (rgba_image[:, :, 1].astype(np.uint32) << 8) |   # Green
        (rgba_image[:, :, 0].astype(np.uint32))          # Red
    )
    
    return img_packed

def update_on_selection(attr, old, new_indices):
    if new_indices:
        new_indices = new_indices[0:num_images]

        # Extract the corresponding x, y coordinates from the source
        selected_x_indices = data_source_points.data['umap_x'][new_indices].tolist()
        selected_y_indices = data_source_points.data['umap_y'][new_indices].tolist()
        file_names = np.array(idxs)[new_indices]

        # Update all images
        new_images = []
        new_x_indices, new_y_indices = [], []
        for i, img in enumerate(images):
            if i <= len(new_indices)-1:
                file_id = file_names[i]
                spectrogram_file_path = os.path.join(spectrogram_folder, file_id.replace('.flac', '.png'))
                #if os.path.join(spectrogram_file_path):
                new_img = load_image_as_rgba(spectrogram_file_path)
                """else:
                    spectrogram = load_audio_file_and_compute_spectrogram(os.path.join(audio_folder, df[df.source_filename == file_id].parent_name.item(), file_id))
                    fig, ax = plt.subplots(figsize=(4,3))
                    ax.imshow(spectrogram, interpolation='nearest', aspect="auto", origin="lower", cmap=cmap)
                    save_borderless(spectrogram_file_path, fig=fig)
                    plt.close(fig)
                    new_img = convert_grayscale_to_rgba_with_cmap(spectrogram, cmap)"""

                # Add jitter to the x and y positions
                jitter_x = random.uniform(-jitter_range, jitter_range)
                jitter_y = random.uniform(-jitter_range, jitter_range)

                new_x_indices.append(selected_x_indices[i] + jitter_x)
                new_y_indices.append(selected_y_indices[i] + jitter_y)
            else:
                new_img = img.copy()
                view = new_img.view(dtype=np.uint8).reshape((W, H, 4))
                view[:, :, 3] = 0
                new_x_indices.append(0)
                new_y_indices.append(0)
            new_images.append(new_img)

        new_x_indices = [x - W / 2 for x in new_x_indices]
        new_y_indices = [y - H / 2 for y in new_y_indices]

        # Update the source data
        data_source_images.data = {
            'image': new_images,
            'x': new_x_indices,
            'y': new_y_indices,
            'dw': dw_values,
            'dh': dh_values,
            'source_filename': file_names
        }

        if save_indices:
            print()
            print('[INFO] Save indices...')
            output_folder = os.path.join(data_root_path, 'Explorer', f'{datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}')
            os.makedirs(output_folder, exist_ok=True)
            np.save(os.path.join(output_folder, 'file_names.npy'), file_names)
            for file_name in file_names:
                file_name = file_name.replace('.flac', '.png')
                print(f'[INFO] Copying {file_name}')
                shutil.copyfile(os.path.join(spectrogram_folder, file_name), os.path.join(output_folder, file_name))
            print('[INFO] Done.')


points_fig_renderer.data_source.selected.on_change('indices', update_on_selection)
fig1.legend.location = "top_left"
fig1.legend.title = "label"
fig1.legend.click_policy = "mute"
