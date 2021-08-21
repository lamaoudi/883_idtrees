
## Tensorboard setup:
# https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


def plot_classes_preds(net, images, targets):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    net.eval()
    output = net(images)

    bboxes = [o['boxes'].to('cpu').detach().numpy() for o in output]
    labels = [o['labels'].to('cpu').detach().numpy() for o in output]
    scores = [o['scores'].to('cpu').detach().numpy() for o in output]
    t_boxes = [t['boxes'].to('cpu').numpy() for t in targets]
    img_id = [t['image_id'].item() for t in targets]
    
    # plot the images in the batch, along with predicted and true labels
#     fig = plt.figure(figsize=(48, 12))
#     fig = plt.figure(figsize=(10,10))#figsize=(48, 12))
    max_imgs = np.minimum(4, len(images))
    f, axs = plt.subplots(1,max_imgs)#,figsize=(5, 15))#*max_imgs))
    for idx in range(max_imgs):
#         ax = fig.add_subplot(1, len(images), idx+1, xticks=[], yticks=[])
#         axs[0] = fig.add_subplot(1, len(images), idx+1, xticks=[], yticks=[])
#         ax = fig.add_subplot(1, 5, idx+1, xticks=[], yticks=[])
        # unnormalize image and change dims : /2 + 0.5
#         plt.imshow(np.transpose(
#             images[idx].to('cpu').detach().numpy(), (1, 2, 0)))

        axs[idx].imshow(np.transpose(
            images[idx].to('cpu').detach().numpy(), (1, 2, 0)))
        for b in bboxes[idx]: #prediction boxes
            # Create a Rectangle patch
            rect = patches.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],
                    linewidth=3,edgecolor='r',facecolor='none')
            # Add the patch to the Axes
#             ax.add_patch(rect)
            axs[idx].add_patch(rect)

        for b in t_boxes[idx]: #target boxes
            # Create a Rectangle patch
            rect = patches.Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],
                    linewidth=2,edgecolor='g',facecolor='none')
            # Add the patch to the Axes
#             ax.add_patch(rect)
            axs[idx].add_patch(rect)

#         ax.set_title(img_id[idx])
        axs[idx].set_title(img_id[idx])

#         if idx==3:
#             break
#     return fig
    return f

def evaluate():
    """
    @author: Dylan Stewart
    updated: 04/16/2020

        input variables:
            GroundTruthBox - numpy array [x y width height]
            DetectionBox   - numpy array [x y width height]

        output:
            score - float
    to use this code:    
        from RandCrowns import halo_parameters
        from RandCrowns import RandNeon

        *if you want to see the plots of the halos
        par = halo_parameters()
        par['im'] = (plot you are using 200x200 for IDTrees Competition)
        score = RandNeon(GroundTruthBox,DetectionBox,par)
        this will give you the score and plot the ground truth, inner, outer,
        and edge halos

        *to run without the plots (faster for large evaluations)
        par = halo_parameters()
        score = RandNeon(GroundTruthBox,DetectionBox,par)
        this will return the score without plotting anything

    """
#     par = halo_parameters()
#     par['im'] = (plot you are using 200x200 for IDTrees Competition)
#     score = RandNeon(GroundTruthBox,DetectionBox,par)
        
def compute_F1():
    pass

"""
def to_tensorboard(writer, loss, F1_score, images):
#     writer.add_image('my_image', img, 0)
#     writer.add_images(image_batch)
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # show images
    matplotlib_imshow(img_grid, one_channel=True)

    # write to tensorboard
    writer.add_image('four_fashion_mnist_images', img_grid)
"""

"""
running_loss = 0.0
for epoch in range(1):  # loop over the dataset multiple times

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:    # every 1000 mini-batches...

            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 1000,
                            epoch * len(trainloader) + i)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(net, inputs, labels),
                            global_step=epoch * len(trainloader) + i)
            running_loss = 0.0
print('Finished Training')
            
"""