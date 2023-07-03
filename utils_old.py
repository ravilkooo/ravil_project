import matplotlib.ticker as ticker
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn


def test_i_sample_old(model, seismic_i, labels_cat_i, show_plots = False):
    resized_im = np.repeat(seismic_i, [(1 + i // 254) for i in range(255)], axis=1)
    resized_lbl = np.repeat(labels_cat_i, [(1 + i // 254) for i in range(255)], axis=1)

    im_parts = np.zeros((4, 256, 256))
    lbl_parts = np.zeros((4, 256, 256, 6))

    im_parts[0] = resized_im[0:256]
    im_parts[1] = resized_im[149:405]
    im_parts[2] = resized_im[298:554]
    im_parts[3] = resized_im[445:701]

    lbl_parts[0] = resized_lbl[0:256]
    lbl_parts[1] = resized_lbl[149:405]
    lbl_parts[2] = resized_lbl[298:554]
    lbl_parts[3] = resized_lbl[445:701]
    lbl_parts = np.moveaxis(lbl_parts, -1, 1)

    output_parts = np.zeros((4, 6, 256, 256))
    for i, output_part in enumerate(output_parts):
        _t = torch.from_numpy(np.expand_dims(np.expand_dims(im_parts[i], axis=0), axis=0)).float()
        # print(f'_t.shape = {_t.shape}')
        output_parts[i:i+1] = nn.Softmax(dim=1)(model(_t)).detach().numpy()
        # print(f'output_parts[i].shape = {output_parts[i].shape}')

    if show_plots:
        fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(6,3))
        for i, output_part in enumerate(output_parts):
            axs[0][i % 4].imshow(im_parts[i,:,:], cmap='gray')
            axs[1][i % 4].imshow(np.argmax(lbl_parts[i], axis=0), cmap='jet', vmin=0, vmax=5)
            axs[2][i % 4].imshow(np.argmax(output_part, axis=0), cmap='jet', vmin=0, vmax=5)
        for i in range(3):
            for j in range(4):
                axs[i][j].xaxis.set_major_locator(ticker.NullLocator())
                axs[i][j].yaxis.set_major_locator(ticker.NullLocator())
        plt.show()

    assert output_parts.shape == lbl_parts.shape, f'Shapes are not equal: {output_parts.shape} != {lbl_parts.shape}'
    full_output = np.zeros(resized_lbl.shape)
    output_parts = np.moveaxis(output_parts, 1, -1)

    full_output[0:149] = output_parts[0][0:149]
    full_output[149:256] = (output_parts[0][149:256] + output_parts[1][0:107]) / 2
    full_output[256:298] = output_parts[1][107:149]
    full_output[298:405] = (output_parts[1][149:256] + output_parts[2][0:107]) / 2
    full_output[405:445] = output_parts[2][107:147]
    full_output[445:554] = (output_parts[2][147:256] + output_parts[3][0:109]) / 2
    full_output[554:701] = output_parts[3][109:256]

    if show_plots:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8,6))
        axs[0].imshow(resized_im[:,:255], cmap='gray')
        axs[1].imshow(np.argmax(resized_lbl[:,:255], axis=-1), cmap='jet', vmin=0, vmax=5)
        axs[2].imshow(np.argmax(full_output[:,:255], axis=-1), cmap='jet', vmin=0, vmax=5)
        for i in range(3):
            axs[i].xaxis.set_major_locator(ticker.NullLocator())
            axs[i].yaxis.set_major_locator(ticker.NullLocator())
        # print(resized_im.shape, resized_lbl.shape, full_output.shape)
        plt.show()
    return resized_im[:,:255], resized_lbl[:,:255], full_output[:,:255]


def test_x_sample_old(model, seismic_x, labels_cat_x, show_plots = False):
    resized_im = np.repeat(seismic_x, [(1 + i // 254) for i in range(255)], axis=1)
    resized_lbl = np.repeat(labels_cat_x, [(1 + i // 254) for i in range(255)], axis=1)

    im_parts = np.zeros((2, 256, 256))
    lbl_parts = np.zeros((2, 256, 256, 6))

    im_parts[0] = resized_im[0:256]
    im_parts[1] = resized_im[145:401]

    lbl_parts[0] = resized_lbl[0:256]
    lbl_parts[1] = resized_lbl[145:401]
    lbl_parts = np.moveaxis(lbl_parts, -1, 1)

    output_parts = np.zeros((2, 6, 256, 256))
    for i, output_part in enumerate(output_parts):
        _t = torch.from_numpy(np.expand_dims(np.expand_dims(im_parts[i], axis=0), axis=0)).float()
        output_parts[i:i+1] = nn.Softmax(dim=1)(model(_t)).detach().numpy()

    if show_plots:
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(6,3))
        for i, output_part in enumerate(output_parts):
            axs[0][i % 2].imshow(im_parts[i,:,:], cmap='gray')
            axs[1][i % 2].imshow(np.argmax(lbl_parts[i], axis=0), cmap='jet', vmin=0, vmax=5)
            axs[2][i % 2].imshow(np.argmax(output_part, axis=0), cmap='jet', vmin=0, vmax=5)
        for i in range(3):
            for j in range(2):
                axs[i][j].xaxis.set_major_locator(ticker.NullLocator())
                axs[i][j].yaxis.set_major_locator(ticker.NullLocator())
        plt.show()

    assert output_parts.shape == lbl_parts.shape, f'Shapes are not equal: {output_parts.shape} != {lbl_parts.shape}'
    full_output = np.zeros(resized_lbl.shape)
    output_parts = np.moveaxis(output_parts, 1, -1)

    full_output[0:145] = output_parts[0][0:145]
    full_output[145:256] = (output_parts[0][145:256] + output_parts[1][0:111]) / 2
    full_output[256:401] = output_parts[1][111:256]
    
    if show_plots:
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8,6))
        axs[0].imshow(resized_im[:,:255], cmap='gray')
        axs[1].imshow(np.argmax(resized_lbl[:,:255], axis=-1), cmap='jet', vmin=0, vmax=5)
        axs[2].imshow(np.argmax(full_output[:,:255], axis=-1), cmap='jet', vmin=0, vmax=5)
        for i in range(3):
            axs[i].xaxis.set_major_locator(ticker.NullLocator())
            axs[i].yaxis.set_major_locator(ticker.NullLocator())
        # print(resized_im.shape, resized_lbl.shape, full_output.shape)
        plt.show()
    return resized_im[:,:255], resized_lbl[:,:255], full_output[:,:255]