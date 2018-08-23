import os
import torch

import matplotlib
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import numpy as np


def plotField(out, tar, flags, loss, mconf, filename=None, save=False,
        plotGraphs=True, plotPres=True, plotVel=True, plotDiv=True):

    target = tar.clone()
    output_p = out[0].clone()
    output_U = out[1].clone()
    p_out = output_p
    p_tar = target[:,0].unsqueeze(1)
    U_norm_out = torch.zeros_like(p_out)
    U_norm_tar = torch.zeros_like(p_tar)

    Ux_out = output_U[:,0].unsqueeze(1)
    Ux_tar = target[:,1].unsqueeze(1)
    Uy_out = output_U[:,1].unsqueeze(1)
    Uy_tar = target[:,2].unsqueeze(1)
    torch.norm(output_U, p=2, dim=1, keepdim=True, out=U_norm_out)
    torch.norm(target[:,1:3], p=2, dim=1, keepdim=True, out=U_norm_tar)

    div = out[2].clone()

    err_p = (p_out - p_tar)**2
    err_Ux = (Ux_out - Ux_tar)**2
    err_Uy = (Uy_out - Uy_tar)**2
    err_div = (div)**2

    max_val_p = np.maximum(torch.max(p_tar).cpu().data.numpy(), \
                         torch.max(p_out).cpu().data.numpy() )
    min_val_p = np.minimum(torch.min(p_tar).cpu().data.numpy(), \
                         torch.min(p_out).cpu().data.numpy())
    max_val_Ux = np.maximum(torch.max(Ux_out).cpu().data.numpy(), \
                         torch.max(Ux_tar).cpu().data.numpy() )
    min_val_Ux = np.minimum(torch.min(Ux_out).cpu().data.numpy(), \
                         torch.min(Ux_tar).cpu().data.numpy())
    max_val_Uy = np.maximum(torch.max(Uy_out).cpu().data.numpy(), \
                         torch.max(Uy_tar).cpu().data.numpy() )
    min_val_Uy = np.minimum(torch.min(Uy_out).cpu().data.numpy(), \
                         torch.min(Uy_tar).cpu().data.numpy())
    max_val_Unorm = np.maximum(torch.max(U_norm_out).cpu().data.numpy(), \
                         torch.max(U_norm_tar).cpu().data.numpy() )
    min_val_Unorm = np.minimum(torch.min(U_norm_out).cpu().data.numpy(), \
                         torch.min(U_norm_tar).cpu().data.numpy() )
    max_err_p = torch.max(err_p).cpu().data.numpy()
    max_err_Ux = torch.max(err_Ux).cpu().data.numpy()
    max_err_Uy = torch.max(err_Uy).cpu().data.numpy()

    max_div = torch.max(div).cpu().data.numpy()
    min_div = torch.min(div).cpu().data.numpy()

    p_tar_line = p_tar.clone()
    p_out_line = p_out.clone()
    mask = flags.eq(2)
    p_tar.masked_fill_(mask, 100)
    p_out.masked_fill_(mask, 100)
    Ux_tar.masked_fill_(mask, 0)
    Ux_out.masked_fill_(mask, 0)
    Uy_tar.masked_fill_(mask, 0)
    Uy_out.masked_fill_(mask, 0)
    U_norm_tar.masked_fill_(mask, 100)
    U_norm_out.masked_fill_(mask, 100)
    div.masked_fill_(mask, 100)

    err_p.masked_fill_(mask, 100)
    err_Ux.masked_fill_(mask, 100)
    err_Uy.masked_fill_(mask, 100)
    err_div.masked_fill_(mask, 100)

    p_tar_line_np =torch.squeeze(p_tar_line.cpu()).data.numpy()
    p_out_line_np =torch.squeeze(p_out_line.cpu()).data.numpy()
    p_tar_np =torch.squeeze(p_tar.cpu()).data.numpy()
    p_out_np =torch.squeeze(p_out.cpu()).data.numpy()
    Ux_tar_np =torch.squeeze(Ux_tar.cpu()).data.numpy()
    Ux_out_np =torch.squeeze(Ux_out.cpu()).data.numpy()
    Uy_tar_np =torch.squeeze(Uy_tar.cpu()).data.numpy()
    Uy_out_np =torch.squeeze(Uy_out.cpu()).data.numpy()
    U_norm_tar_np =torch.squeeze(U_norm_tar.cpu()).data.numpy()
    U_norm_out_np =torch.squeeze(U_norm_tar.cpu()).data.numpy()
    div_np =torch.squeeze(div).cpu().data.numpy()
    err_p_np =torch.squeeze(err_p.cpu()).data.numpy()
    err_Ux_np =torch.squeeze(err_Ux.cpu()).data.numpy()
    err_Uy_np =torch.squeeze(err_Uy.cpu()).data.numpy()
    err_div_np =torch.squeeze(err_div.cpu()).data.numpy()

    title_list = []
    numLoss = 0
    if mconf['pL2Lambda'] > 0:
        numLoss +=1

    if mconf['divL2Lambda'] > 0:
        numLoss +=1

    if mconf['pL1Lambda'] > 0:
        numLoss +=1

    if mconf['divL1Lambda'] > 0:
        numLoss +=1

    if ('divLongTermLambda' in mconf) and mconf['divLongTermLambda'] > 0:
        numLoss +=1

    if mconf['pL2Lambda'] > 0:
        title_list.append(str(mconf['pL2Lambda']) + ' * L2(p)')

    if mconf['divL2Lambda'] > 0:
        title_list.append(str(mconf['divL2Lambda']) + ' * L2(div)')

    if mconf['pL1Lambda'] > 0:
        title_list.append(str(mconf['pL1Lambda']) + ' * L1(p)')

    if mconf['divL1Lambda'] > 0:
        title_list.append(str(mconf['divL1Lambda']) + ' * L1(div)')

    if ('divLongTermLambda' in mconf) and (mconf['divLongTermLambda'] > 0):
        title_list.append(str(mconf['divLongTermLambda']) + ' * L2(LongTermDiv)')

    title = ''
    for string in range(0, numLoss - 1):
        title += title_list[string] + ' + '
    title += title_list[numLoss-1]

    my_cmap = cm.jet
    my_cmap.set_over('white')
    my_cmap.set_under('white')

    nrow = 0
    height_ratios = []
    if plotGraphs:
        nrow +=1
        height_ratios.append(1)
    if plotPres:
        nrow += 1
        height_ratios.append(1)
    if plotVel:
        nrow += 1
        height_ratios.append(1)
    if plotDiv:
        nrow += 1
        height_ratios.append(1)

    ncol = 3
    matplotlib.rc('text')
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(nrow, ncol,
                     width_ratios=[1,1,1],
                     height_ratios=height_ratios,
                     wspace=0.2, hspace=0.2, top=0.9, bottom=0.01,
                     left=0.01, right=0.99)
    fig.suptitle('FluidNet output for loss = ' + title )

    it_row = 0

    if plotPres:
        s = np.linspace(0, 127)
        f_s = 60

        ax = plt.subplot(gs[it_row,0])
        ax.set_title('P_target')
        ax.axis('off')
        ax.imshow(p_tar_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[min_val_p,max_val_p])
        ax.plot(s, [f_s for i in range(len(s))])

        ax = plt.subplot(gs[it_row, 1])
        ax.set_title('P_predicted')
        ax.axis('off')
        ax.imshow(p_out_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[min_val_p,max_val_p])

        ax = plt.subplot(gs[it_row, 2])
        ax.set_title('error P')
        ax.axis('off')
        ax.imshow(err_p_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[0,max_err_p])
        it_row += 1

        P_out = p_out_line_np[f_s,:]
        P_tar = p_tar_line_np[f_s,:]
        ax = plt.subplot(gs[it_row,1])
        ax.set_title('P')
        ax.plot(P_out, label = 'Output pressure')
        ax.plot(P_tar, label = 'Target pressure')
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        ax.set_aspect((x1-x0)/(y1-y0))
        ax.legend()
        ax.set_ylabel('Pressure')
        it_row += 1

    if plotVel:
        skip = 4
        scale = 0.1
        scale_units = 'xy'
        angles = 'xy'
        headwidth = 3
        headlength = 3
        Ux_tar_np_adm = Ux_tar_np / np.max(np.sqrt(Ux_tar_np **2 + Uy_tar_np **2))
        Uy_tar_np_adm = Uy_tar_np / np.max(np.sqrt(Ux_tar_np **2 + Uy_tar_np **2))
        Ux_out_np_adm = Ux_out_np / np.max(np.sqrt(Ux_out_np **2 + Uy_out_np **2))
        Uy_out_np_adm = Uy_out_np / np.max(np.sqrt(Ux_out_np **2 + Uy_out_np **2))

        ax = plt.subplot(gs[it_row, 0])
        ax.set_title('|U|_target')
        ax.axis('off')
        X, Y = np.linspace(0, 127, num=128), np.linspace(0, 127, num=128)
        ax.imshow(U_norm_tar_np, cmap=my_cmap, origin='lower',
                interpolation='none', clim=[min_val_Unorm,max_val_Unorm])
        ax.quiver(X[::skip], Y[::skip],
                Ux_tar_np_adm[::skip, ::skip], Uy_tar_np_adm[::skip, ::skip],
                scale_units=scale_units,
                angles=angles,
                headwidth=headwidth, headlength=headlength, scale=scale,
                color='pink')

        ax = plt.subplot(gs[it_row, 1])
        ax.set_title('|U|_predicted')
        ax.axis('off')
        ax.imshow(U_norm_out_np, cmap=my_cmap, origin='lower',
                interpolation='none', clim=[min_val_Unorm,max_val_Unorm])
        ax.quiver(X[::skip], Y[::skip],
                Ux_out_np_adm[::skip, ::skip], Uy_out_np_adm[::skip, ::skip],
                scale_units=scale_units, 
                angles=angles,
                headwidth=headwidth, headlength=headlength, scale=scale,
                color='pink')

        ax = plt.subplot(gs[it_row, 2])
        ax.set_title('error Ux')
        ax.axis('off')
        ax.imshow(err_Ux_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[0,max_err_Ux])
        it_row += 1

    if plotDiv:
        ax = plt.subplot(gs[it_row, 0])
        ax.set_title('div at output')
        ax.axis('off')
        ax.imshow(div_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[min_div,max_div])

        ax = plt.subplot(gs[it_row, 1])
        ax.set_title('div error')
        ax.axis('off')
        ax.imshow(err_div_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[0,max_div**2])

        ax = plt.subplot(gs[it_row, 2])
        ax.set_title('Losses')
        ax.axis('off')
        for y, loss, val in zip(np.linspace(0.2,0.8,5),
                                 ['Total' ,'L2(p)', 'L2(div)', 'L1(p)', 'L1(div)'],
                                 loss):
            ax.text(0.2, y,  ('Loss {:} : ').format(loss) \
                    + ('{:.6f}').format(val), fontsize=12)
        it_row += 1

    #fig.colorbar(imP, cax=cbar_ax_p, orientation='vertical')
    #cbar_ax_U = fig.add_axes([0.375, 0.45, 0.01, 0.33])
    #fig.colorbar(imU, cax=cbar_ax_U, orientation='vertical')
    #fig.set_size_inches((11, 11), forward=False)
    if save:
        fig.savefig(filename)
    else:
        plt.show(block=True)

