import numpy as np
import torch
import cv2
from .chamfer_distance import cd_score


# def make_inter_graph(v2d1, v2d2, topo1, topo2, match12):
#     valid = (match12 != -1)
#     marked2 = np.zeros(len(v2d2)).astype(bool)
#     # print(match12[valid])
#     marked2[match12[valid]] = True

#     id1toh, id2toh = np.zeros(len(v2d1)), np.zeros(len(v2d2))
#     id1toh[valid] = np.arange(np.sum(valid))
#     id2toh[match12[valid]] = np.arange(np.sum(valid))
#     id1toh[np.invert(valid)] = np.arange(np.sum(1 - valid)) + np.sum(valid)
#     # print(marked2)
#     id2toh[np.invert(marked2)] = len(v2d1) + np.arange(np.sum(np.invert(marked2)))

#     id1toh = id1toh.astype(int)
#     id2toh = id2toh.astype(int)

#     tot_len = len(v2d1) + np.sum(np.invert(marked2))

#     vin1 = v2d1[valid][:]
#     vin2 = v2d2[match12[valid]][:]
#     vh = 0.5 * (vin1 + vin2)
#     vh = np.concatenate((vh, v2d1[np.invert(valid)], v2d2[np.invert(marked2)]), axis=0)

#     topoh = [[] for ii in range(tot_len)]


#     for node in range(len(topo1)):
        
#         for nb in topo1[node]:
#             if int(id1toh[nb]) not in topoh[id1toh[node]]:
#                 topoh[id1toh[node]].append(int(id1toh[nb]))


#     for node in range(len(topo2)):
#         for nb in topo2[node]:
#             if int(id2toh[nb]) not in topoh[id2toh[node]]:
#                 topoh[id2toh[node]].append(int(id2toh[nb]))

#     return vh, topoh


# def make_inter_graph_valid(v2d1, v2d2, topo1, topo2, match12):
#     valid = (match12 != -1)
#     marked2 = np.zeros(len(v2d2)).astype(bool)
#     # print(match12[valid])
#     marked2[match12[valid]] = True

#     id1toh, id2toh = np.zeros(len(v2d1)), np.zeros(len(v2d2))
#     id1toh[valid] = np.arange(np.sum(valid))
#     id2toh[match12[valid]] = np.arange(np.sum(valid))
#     id1toh[np.invert(valid)] = np.arange(np.sum(1 - valid)) + np.sum(valid)
#     # print(marked2)
#     id2toh[np.invert(marked2)] = len(v2d1) + np.arange(np.sum(np.invert(marked2)))

#     id1toh = id1toh.astype(int)
#     id2toh = id2toh.astype(int)

#     tot_len = len(v2d1) + np.sum(np.invert(marked2))

#     vin1 = v2d1[valid][:]
#     vin2 = v2d2[match12[valid]][:]
#     vh = 0.5 * (vin1 + vin2)
#     # vh = np.concatenate((vh, v2d1[np.invert(valid)], v2d2[np.invert(marked2)]), axis=0)

#     # topoh = [[] for ii in range(tot_len)]
#     topoh = [[] for ii in range(np.sum(valid))]

#     for node in range(len(topo1)):
#         if not valid[node]:
#             continue
#         for nb in topo1[node]:
#             if int(id1toh[nb]) not in topoh[id1toh[node]]:
#                 if valid[nb]:
#                     topoh[id1toh[node]].append(int(id1toh[nb]))


#     for node in range(len(topo2)):
#         if not marked2[node]:
#             continue
#         for nb in topo2[node]:
#             if int(id2toh[nb]) not in topoh[id2toh[node]]:
#                 if marked2[nb]:
#                     topoh[id2toh[node]].append(int(id2toh[nb]))

#     return vh, topoh



def visualize(dict):
    # print(dict['keypoints0'].size(), flush=True)
    img1 = ((dict['image0'][0].permute(1, 2, 0).float().numpy() + 1.0) * 255 / 2).astype(np.uint8).copy()
    img2 = ((dict['image1'][0].permute(1, 2, 0).float().numpy() + 1.0) * 255 / 2).astype(np.uint8).copy()
    original_target = ((dict['imaget'][0].permute(1, 2, 0).float().numpy() + 1.0) * 255 / 2).astype(np.uint8).copy()
    # img1p = ((dict['image0'].permute(1, 2, 0).float().numpy() + 1.0) * 255 / 2).astype(int).copy()
    # img2p = ((dict['image1'].permute(1, 2, 0).float().numpy() + 1.0) * 255 / 2).astype(int).copy()

    # img1[:, :, 0] += 255
    # img1[:, :, 1] += 180
    # img1[:, :, 2] += 180
    # img1[img1 > 255] = 255

    # img2[:, :, 0] += 255
    # img2[:, :, 1] += 180
    # img2[:, :, 2] += 180
    # img2[img2 > 255] = 255
    
    # img1p[:, :, 0] += 255
    # img1p[:, :, 1] += 180
    # img1p[:, :, 2] += 180
    # img1p[img1p > 255] = 255
    
    # img2p[:, :, 0] += 255
    # img2p[:, :, 1] += 180
    # img2p[:, :, 2] += 180
    # img2p[img2p > 255] = 255

    # img1, img2, img1p, img2p = img1.astype(np.uint8), img2.astype(np.uint8), img1p.astype(np.uint8), img2p.astype(np.uint8)
    r0 = dict['r0'][0].cpu().numpy().astype(int) 
    r1 = dict['r1'][0].cpu().numpy().astype(int) 

    source0_warp = dict['keypoints0t'][0].cpu().numpy().astype(int)
    source2_warp = dict['keypoints1t'][0].cpu().numpy().astype(int)
    source0 = dict['keypoints0'][0].cpu().numpy().astype(int)
    source2 = dict['keypoints1'][0].cpu().numpy().astype(int)
    source0_topo = dict['topo0'][0]
    # print(len(dict['topo0']))
    source2_topo = dict['topo1'][0]
    visible01 = dict['vb0'][0].cpu().numpy().astype(int)
    visible21 = dict['vb1'][0].cpu().numpy().astype(int)

    # corr01 = dict['m01'][0].cpu().numpy().astype(int)
    # corr10 = dict['m10'][0].cpu().numpy().astype(int)

    # canvas = np.zeros_like(img1) + 255

    # source0_warp2 = source0 + motion01 // 2
    # source2_warp2 = source2 + motion21 // 2

    # for node, nbs in enumerate(source0_topo):
    #     for nb in nbs:
    #         # print([source0_warp[nb][0], source0_warp[nb][1]])
    #         cv2.line(canvas, [source0_warp[node][0], source0_warp[node][1]], [source0_warp[nb][0], source0_warp[nb][1]], [0, 0, 0], 2)
    # for node, nbs in enumerate(source2_topo):
    #     for nb in nbs:
    #         cv2.line(canvas, [source2_warp[node][0], source2_warp[node][1]], [source2_warp[nb][0], source2_warp[nb][1]], [0, 0, 0], 2)


    # canvas6 = np.zeros_like(img1) + 255


    # for node, nbs in enumerate(source0_topo):
    #     for nb in nbs:
    #         # print([source0_warp[nb][0], source0_warp[nb][1]])
    #         cv2.line(canvas6, [source0_warp2[node][0], source0_warp2[node][1]], [source0_warp2[nb][0], source0_warp2[nb][1]], [0, 0, 0], 2)
    # for node, nbs in enumerate(source2_topo):
    #     for nb in nbs:
    #         cv2.line(canvas6, [source2_warp2[node][0], source2_warp2[node][1]], [source2_warp2[nb][0], source2_warp2[nb][1]], [0, 0, 0], 2)

    canvas2 = np.zeros_like(img1) + 255

    # source0_warp = source0 + motion01
    # source2_warp = source2 + motion21

    for node, nbs in enumerate(source0_topo):
        for nb in nbs:
            # if visible01[node] and visible01[nb]:
            cv2.line(canvas2, [source0_warp[node][0], source0_warp[node][1]], [source0_warp[nb][0], source0_warp[nb][1]], [0, 0, 0], 2)
    for node, nbs in enumerate(source2_topo):
        for nb in nbs:
            # if visible21[node] and visible21[nb]:
            cv2.line(canvas2, [source2_warp[node][0], source2_warp[node][1]], [source2_warp[nb][0], source2_warp[nb][1]], [0, 0, 0], 2)

    

    # canvas2
    # black_threshold = 255 // 2
    # img1_sketch = rgb2sketch(img1, black_threshold)
    # img2_sketch = rgb2sketch(img2, black_threshold)

    # img1_sketch = img1_sketch.unsqueeze(0)
    # img2_sketch = img2_sketch.unsqueeze(0)

    # CD = ChamferDistance2dMetric()
    # cd = CD(img1_sketch,img2_sketch)
    canvases = [np.zeros_like(img1) + 255, np.zeros_like(img1) + 255, np.zeros_like(img1) + 255, np.zeros_like(img1) + 255]
    canvas5 = np.zeros_like(img1) + 255
    # canvas7 = np.zeros_like(img1) + 255
    # canvas8 = np.zeros_like(img1) + 255


    # source0_warp = source0 + motion01
    # source2_warp = source2 + motion21
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



    canvas3 = np.zeros_like(img1) + 255

    for node, nbs in enumerate(source0_topo):
        for nb in nbs:
            cv2.line(canvas3, [source0[node][0], source0[node][1]], [source0[nb][0], source0[nb][1]], [255, 180, 180], 2)
    for node, nbs in enumerate(source2_topo):
        for nb in nbs:
            cv2.line(canvas3, [source2[node][0], source2[node][1]], [source2[nb][0], source2[nb][1]], [180, 180, 255], 2)

    #canvas6, canvas5, canvas, 
    # im_h = cv2.hconcat([canvas3, original_target, canvas2, canvas5])
    im_h = cv2.hconcat([img1] + canvases + [img2])
    cd = cd_score(canvas5.copy(), original_target.copy()) * 1e5

    # cv2.putText(im_h, str(cd), \
    #     (720, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)


    
    return im_h, cd
