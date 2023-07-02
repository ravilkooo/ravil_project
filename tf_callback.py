import matplotlib.pyplot as plt
import itertools
import numpy as np
import tensorflow as tf
import io
import torch
import matplotlib.ticker as ticker
import torch.nn as nn


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # print(f'images.shape = {images.shape}')
    # print(f'output.shape = {output.shape}')
    
    output_soft = nn.functional.softmax(output, dim=1)
    probs_tensor, preds_tensor = torch.max(output_soft, 1)
    
    # print(f'probs_tensor.shape = {probs_tensor.shape}')
    # print(f'preds_tensor.shape = {preds_tensor.shape}')
    
    return probs_tensor, preds_tensor


def plot_classes_preds(net, images, labels, epoch):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    probs, preds = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    batch_size = preds.shape[0]

    fig, axs = plt.subplots(nrows=3, ncols=batch_size, figsize=(10, 5))
    
    for idx in np.arange(batch_size):
        axs[0][idx].imshow(images[idx,0,:,:], cmap='gray')
        axs[1][idx].imshow(np.argmax(labels[idx],axis=0), cmap='jet', vmin=0, vmax=5)
        axs[2][idx].imshow(preds[idx], cmap='jet', vmin=0, vmax=5)
        for j in range(3):
            axs[j][idx].xaxis.set_major_locator(ticker.NullLocator())
            axs[j][idx].yaxis.set_major_locator(ticker.NullLocator())
    fig.set_layout_engine('tight')
    fig.suptitle(f'epoch #{epoch}')
    return fig


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

'''
# Use the model to predict the values from the validation dataset.
test_pred_raw = model.predict(test_images)
test_pred = np.argmax(test_pred_raw, axis=1)

# Calculate the confusion matrix.
cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
# Log the confusion matrix as an image summary.
'''    
'''
def log_confusion_matrix(cm, epoch, class_names):
    figure = plot_confusion_matrix(cm, class_names=class_names)
    cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)
'''