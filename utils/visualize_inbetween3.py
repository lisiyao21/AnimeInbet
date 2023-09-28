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
    motion01 = dict['motion0'][0].cpu().numpy().astype(int) 
    motion21 = dict['motion1'][0].cpu().numpy().astype(int) 

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

  #   canvas2 = np.zeros_like(img1) + 255

  # ##  print('huala<<<', source0_warp.mean(), source2_warp.mean(), flush=True)

  #   # source0_warp = source0 + motion01
  #   # source2_warp = source2 + motion21

  #   for node, nbs in enumerate(source0_topo):
  #       for nb in nbs:
  #           # if visible01[node] and visible01[nb]:
  #           cv2.line(canvas2, [source0_warp[node][0], source0_warp[node][1]], [source0_warp[nb][0], source0_warp[nb][1]], [0, 0, 0], 2)
  #   for node, nbs in enumerate(source2_topo):
  #       for nb in nbs:
  #           # if visible21[node] and visible21[nb]:
  #           cv2.line(canvas2, [source2_warp[node][0], source2_warp[node][1]], [source2_warp[nb][0], source2_warp[nb][1]], [0, 0, 0], 2)

    

    # canvas2
    # black_threshold = 255 // 2
    # img1_sketch = rgb2sketch(img1, black_threshold)
    # img2_sketch = rgb2sketch(img2, black_threshold)

    # img1_sketch = img1_sketch.unsqueeze(0)
    # img2_sketch = img2_sketch.unsqueeze(0)

    # CD = ChamferDistance2dMetric()
    # cd = CD(img1_sketch,img2_sketch)
    canvas5 = np.zeros_like(img1) + 255

    # source0_warp = source0 + motion01
    # source2_warp = source2 + motion21

  ##  print('gulaa>>>', visible01.mean(), visible21.mean(), flush=True)

    for node, nbs in enumerate(source0_topo):
        for nb in nbs:
            if visible01[node] and visible01[nb]:
                cv2.line(canvas5, [source0_warp[node][0], source0_warp[node][1]], [source0_warp[nb][0], source0_warp[nb][1]], [0, 0, 0], 2)
    for node, nbs in enumerate(source2_topo):
        for nb in nbs:
            if visible21[node] and visible21[nb]:
                cv2.line(canvas5, [source2_warp[node][0], source2_warp[node][1]], [source2_warp[nb][0], source2_warp[nb][1]], [0, 0, 0], 2)



    # canvas3 = np.zeros_like(img1) + 255
    

    # for node, nbs in enumerate(source0_topo):
    #     for nb in nbs:
    #         cv2.line(canvas3, [source0[node][0], source0[node][1]], [source0[nb][0], source0[nb][1]], [255, 180, 180], 2)
    # for node, nbs in enumerate(source2_topo):
    #     for nb in nbs:
    #         cv2.line(canvas3, [source2[node][0], source2[node][1]], [source2[nb][0], source2[nb][1]], [180, 180, 255], 2)

    # canvas_corr1 = np.zeros_like(img1) + 255
    # canvas_corr2 = np.zeros_like(img1) + 255

    # canvas_corr1 = ((dict['image0'][0].permute(1, 2, 0).float().numpy() + 1.0) * 255 / 2).astype(int).copy()
    # canvas_corr2 = ((dict['image1'][0].permute(1, 2, 0).float().numpy() + 1.0) * 255 / 2).astype(int).copy()

    # canvas_corr1[:, :, 0] += 255
    # canvas_corr1[:, :, 1] += 180
    # canvas_corr1[:, :, 2] += 180
    # canvas_corr1[canvas_corr1 > 255] = 255

    # canvas_corr2[:, :, 0] += 255
    # canvas_corr2[:, :, 1] += 180
    # canvas_corr2[:, :, 2] += 180
    # canvas_corr2[canvas_corr2 > 255] = 255

    # canvas_corr1 = canvas_corr1.astype(np.uint8)
    # canvas_corr2 = canvas_corr2.astype(np.uint8)

    # # colors1_gt, colors2_gt = {}, {}
    # colors1_pred, colors2_pred = {}, {}
    # # cross1_pred, cross2_pred = {}, {}
    # id1 = np.arange(len(source0))
    # id2 = np.arange(len(source2))

    # predicted = dict['matches0'].cpu().data.numpy()[0]
    # predicted1 = dict['matches1'].cpu().data.numpy()[0]
    # for index in id1:
    #     color = [np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]
    #         # print(predicted.shape, flush=True)
    #     # if all_matches[index] != -1:
    #     #     colors2_gt[all_matches[index]] = color
    #     if predicted[index] != -1:
    #         colors2_pred[predicted[index]] = color
    #     # else:
    #     #     colors2_pred[predicted[index]] = [0, 0, 0]

    #     # colors1_gt[index] = color if all_matches[index] != -1 else [0, 0, 0]
    #     colors1_pred[index] = color if predicted[index] != -1 else [0, 0, 0]

    #     # if predicted[index] == -1 and colors1_pred[index] != [0, 0, 0]:
    #     #     color = [np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]
    #     #     colors1_pred[index] = [0, 0, 0]
    #     #     colors2_pred.pop(all_matches[index])
    #     # whether predicted correctly
    #     # if predicted[index] != all_matches[index]:
    #     #     cross1_pred[index] = True
    #     #     if predicted[index] != -1:
    #     #         cross2_pred[predicted[index]] = True
        
    # for i, p in enumerate(source0):
    #     ii = id1[i]
    #     # print(ii)
    #     cv2.circle(canvas_corr1, [int(p[0]), int(p[1])], 1, colors1_pred[i], 2)
    #     # if ii in cross1_pred and cross1_pred[ii]:
    #     #     cv2.rectangle(img1p, [int(p[0]-1), int(p[1]-1)], [int(p[0]+1), int(p[1]+1)], colors1_pred[i],-1)
    #     # else:
    #     #     cv2.circle(img1p, [int(p[0]), int(p[1])], 1, colors1_pred[i], 2)
        
    # for ii in id2:
    #     # print(ii)
    #     color = [0, 0, 0]
    #     this_is_umatched = 1
    #     if ii not in colors2_pred:
    #         colors2_pred[ii] = color

    # for i, p in enumerate(source2):
    #     ii = id2[i]
    #     # print(p)
    #     # cv2.circle(img2, [int(p[0]), int( p[1])], 1, colors2_gt[ii], 2)
    #     # if ii in cross2_pred and cross2_pred[ii]:
    #     #     cv2.rectangle(img2p, [int(p[0]-1), int(p[1]-1)], [int(p[0]+1), int(p[1]+1)], colors2_pred[i], -1)
    #     # else:
    #     cv2.circle(canvas_corr2, [int(p[0]), int(p[1])], 1, colors2_pred[i], 2)




    #canvas6, canvas5, canvas, 
    im_h = cv2.hconcat([canvas5])
    # im_h = canvas5
  ##  print('<<<< mean cavans5: ', canvas5.mean())
    # cd = cd_score(canvas5.copy(), original_target.copy()) * 1e5

    # cv2.putText(im_h, str(cd), \
    #     (720, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)


    
    return im_h
