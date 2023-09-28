import numpy as np
import torch
import cv2



def visvid(dict, inter_frames=1):
    img1 = ((dict['image0'][0].permute(1, 2, 0).float().numpy() + 1.0) * 255 / 2).astype(np.uint8).copy()
    img2 = ((dict['image1'][0].permute(1, 2, 0).float().numpy() + 1.0) * 255 / 2).astype(np.uint8).copy()

    r0 = dict['r0'][0].cpu().numpy()
    r1 = dict['r1'][0].cpu().numpy()

    source0 = dict['keypoints0'][0].cpu().numpy()
    source2 = dict['keypoints1'][0].cpu().numpy()
 
    source0_topo = dict['ntopo0'][0]

    source2_topo = dict['ntopo1'][0]
    ori_source0_topo = dict['topo0'][0]

    ori_source2_topo = dict['topo1'][0]
    visible01 = dict['vb0'][0].cpu().numpy().astype(int)
    visible21 = dict['vb1'][0].cpu().numpy().astype(int)

    canvas1 = np.zeros_like(img1) + 255
    canvas2 = np.zeros_like(img1) + 255

    for node, nbs in enumerate(ori_source0_topo):
        for nb in nbs:
            cv2.line(canvas1, [source0[node][0], source0[node][1]], [source0[nb][0], source0[nb][1]], [0, 0, 0], 2)
    for node, nbs in enumerate(ori_source2_topo):
        for nb in nbs:
            cv2.line(canvas2, [source2[node][0], source2[node][1]], [source2[nb][0], source2[nb][1]], [0, 0, 0], 2)


    canvases = [ np.zeros_like(img1).copy() + 255 for jj in range(inter_frames)  ] 

    for ii in range(len(canvases)):
        source0_warp = (source0 + (ii + 1.0) / (len(canvases) + 1.0) * r0).astype(int)
        source2_warp = (source2 + (1 - (ii + 1.0) / (len(canvases) + 1.0)) * r1).astype(int)
        for node, nbs in enumerate(source0_topo):
            for nb in nbs:
                if visible01[node] and visible01[nb]:
                    cv2.line(canvases[ii], [source0_warp[node][0], source0_warp[node][1]], [source0_warp[nb][0], source0_warp[nb][1]], [0, 0, 0], 2)
        for node, nbs in enumerate(source2_topo):
            for nb in nbs:
                if visible21[node] and visible21[nb]:
                    cv2.line(canvases[ii], [source2_warp[node][0], source2_warp[node][1]], [source2_warp[nb][0], source2_warp[nb][1]], [0, 0, 0], 2)
        # if ii == 15:
          ##  print('hulala>>>>', source0_warp.mean(), source2_warp.mean(), (ii + 1.0) / (len(canvases) + 1.0), (1 - (ii + 1.0) / (len(canvases) + 1.0)), flush=True)
          ##  print(canvases[ii].mean())

    for ii in range(len(canvases)):
        canvases[ii] =  cv2.hconcat([canvas1, canvases[ii]])

    images = [cv2.hconcat([canvas1, canvas1])] + canvases + [cv2.hconcat([canvas2, canvas2])]
    
    return images
