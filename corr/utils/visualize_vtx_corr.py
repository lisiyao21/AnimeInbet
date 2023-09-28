import numpy as np
import torch
import cv2


def make_inter_graph(v2d1, v2d2, topo1, topo2, match12):
    valid = (match12 != -1)
    marked2 = np.zeros(len(v2d2)).astype(bool)
    # print(match12[valid])
    marked2[match12[valid]] = True

    id1toh, id2toh = np.zeros(len(v2d1)), np.zeros(len(v2d2))
    id1toh[valid] = np.arange(np.sum(valid))
    id2toh[match12[valid]] = np.arange(np.sum(valid))
    id1toh[np.invert(valid)] = np.arange(np.sum(1 - valid)) + np.sum(valid)
    # print(marked2)
    id2toh[np.invert(marked2)] = len(v2d1) + np.arange(np.sum(np.invert(marked2)))

    id1toh = id1toh.astype(int)
    id2toh = id2toh.astype(int)

    tot_len = len(v2d1) + np.sum(np.invert(marked2))

    vin1 = v2d1[valid][:]
    vin2 = v2d2[match12[valid]][:]
    vh = 0.5 * (vin1 + vin2)
    vh = np.concatenate((vh, v2d1[np.invert(valid)], v2d2[np.invert(marked2)]), axis=0)

    topoh = [[] for ii in range(tot_len)]


    for node in range(len(topo1)):
        
        for nb in topo1[node]:
            if int(id1toh[nb]) not in topoh[id1toh[node]]:
                topoh[id1toh[node]].append(int(id1toh[nb]))


    for node in range(len(topo2)):
        for nb in topo2[node]:
            if int(id2toh[nb]) not in topoh[id2toh[node]]:
                topoh[id2toh[node]].append(int(id2toh[nb]))

    return vh, topoh


def make_inter_graph_valid(v2d1, v2d2, topo1, topo2, match12):
    valid = (match12 != -1)
    marked2 = np.zeros(len(v2d2)).astype(bool)
    # print(match12[valid])
    marked2[match12[valid]] = True

    id1toh, id2toh = np.zeros(len(v2d1)), np.zeros(len(v2d2))
    id1toh[valid] = np.arange(np.sum(valid))
    id2toh[match12[valid]] = np.arange(np.sum(valid))
    id1toh[np.invert(valid)] = np.arange(np.sum(1 - valid)) + np.sum(valid)
    # print(marked2)
    id2toh[np.invert(marked2)] = len(v2d1) + np.arange(np.sum(np.invert(marked2)))

    id1toh = id1toh.astype(int)
    id2toh = id2toh.astype(int)

    tot_len = len(v2d1) + np.sum(np.invert(marked2))

    vin1 = v2d1[valid][:]
    vin2 = v2d2[match12[valid]][:]
    vh = 0.5 * (vin1 + vin2)
    # vh = np.concatenate((vh, v2d1[np.invert(valid)], v2d2[np.invert(marked2)]), axis=0)

    # topoh = [[] for ii in range(tot_len)]
    topoh = [[] for ii in range(np.sum(valid))]

    for node in range(len(topo1)):
        if not valid[node]:
            continue
        for nb in topo1[node]:
            if int(id1toh[nb]) not in topoh[id1toh[node]]:
                if valid[nb]:
                    topoh[id1toh[node]].append(int(id1toh[nb]))


    for node in range(len(topo2)):
        if not marked2[node]:
            continue
        for nb in topo2[node]:
            if int(id2toh[nb]) not in topoh[id2toh[node]]:
                if marked2[nb]:
                    topoh[id2toh[node]].append(int(id2toh[nb]))

    return vh, topoh



def visualize(dict):
    # print(dict['keypoints0'].size(), flush=True)
    img1 = ((dict['image0'].permute(1, 2, 0).float().numpy() + 1.0) * 255 / 2).astype(int).copy()
    img2 = ((dict['image1'].permute(1, 2, 0).float().numpy() + 1.0) * 255 / 2).astype(int).copy()
    img1p = ((dict['image0'].permute(1, 2, 0).float().numpy() + 1.0) * 255 / 2).astype(int).copy()
    img2p = ((dict['image1'].permute(1, 2, 0).float().numpy() + 1.0) * 255 / 2).astype(int).copy()

    img1[:, :, 0] += 255
    img1[:, :, 1] += 180
    img1[:, :, 2] += 180
    img1[img1 > 255] = 255

    img2[:, :, 0] += 255
    img2[:, :, 1] += 180
    img2[:, :, 2] += 180
    img2[img2 > 255] = 255
    
    img1p[:, :, 0] += 255
    img1p[:, :, 1] += 180
    img1p[:, :, 2] += 180
    img1p[img1p > 255] = 255
    
    img2p[:, :, 0] += 255
    img2p[:, :, 1] += 180
    img2p[:, :, 2] += 180
    img2p[img2p > 255] = 255

    img1, img2, img1p, img2p = img1.astype(np.uint8), img2.astype(np.uint8), img1p.astype(np.uint8), img2p.astype(np.uint8)
    

    # print(v2d1.shape, img1.shape, flush=True)
    v2d1 = dict['keypoints0'].numpy().astype(int)
    v2d2 = dict['keypoints1'].numpy().astype(int)
    topo1 = dict['topo0']
    topo2 = dict['topo1']
    # print(topo1, flush=True)
    # for node, nbs in enumerate(dict['topo0']):
    #     for nb in nbs:
    #         cv2.line(img1, [v2d1[node][0], v2d1[node][1]], [v2d1[nb][0], v2d1[nb][1]], [255, 180, 180], 2)


    id1 = np.arange(len(v2d1))
    id2 = np.arange(len(v2d2))
    all_matches = dict['all_matches'].cpu().int().data.numpy()
    predicted = dict['matches0'].cpu().data.numpy()[0]
    predicted1 = dict['matches1'].cpu().data.numpy()[0]
    
    colors1_gt, colors2_gt = {}, {}
    colors1_pred, colors2_pred = {}, {}
    cross1_pred, cross2_pred = {}, {}

    for index in id1:
        color = [np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]
            # print(predicted.shape, flush=True)
        if all_matches[index] != -1:
            colors2_gt[all_matches[index]] = color
        if predicted[index] != -1:
            colors2_pred[predicted[index]] = color

        colors1_gt[index] = color if all_matches[index] != -1 else [0, 0, 0]
        colors1_pred[index] = color if predicted[index] != -1 else [0, 0, 0]

        # if predicted[index] == -1 and colors1_pred[index] != [0, 0, 0]:
        #     color = [np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]
        #     colors1_pred[index] = [0, 0, 0]
        #     colors2_pred.pop(all_matches[index])
        # whether predicted correctly
        if predicted[index] != all_matches[index]:
            cross1_pred[index] = True
            if predicted[index] != -1:
                cross2_pred[predicted[index]] = True
        
    for i, p in enumerate(v2d1):
        ii = id1[i]
        # print(ii)
        cv2.circle(img1, [int(p[0]), int(p[1])], 1, colors1_gt[i], 2)
        if ii in cross1_pred and cross1_pred[ii]:
            cv2.rectangle(img1p, [int(p[0]-1), int(p[1]-1)], [int(p[0]+1), int(p[1]+1)], colors1_pred[i],-1)
        else:
            cv2.circle(img1p, [int(p[0]), int(p[1])], 1, colors1_pred[i], 2)
        
    for ii in id2:
        # print(ii)
        color = [0, 0, 0]
        this_is_umatched = 1
        if ii not in colors2_gt:
            colors2_gt[ii] = color  
        if ii not in colors2_pred:
            colors2_pred[ii] = color

    for i, p in enumerate(v2d2):
        ii = id2[i]
        # print(p)
        cv2.circle(img2, [int(p[0]), int( p[1])], 1, colors2_gt[ii], 2)
        if ii in cross2_pred and cross2_pred[ii]:
            cv2.rectangle(img2p, [int(p[0]-1), int(p[1]-1)], [int(p[0]+1), int(p[1]+1)], colors2_pred[i], -1)
        else:
            cv2.circle(img2p, [int(p[0]), int(p[1])], 1, colors2_pred[i], 2)

    # print('Unmatched in Img 2: ', , '%')
    # unmatched_all.append(100 - unmatched * 100.0/len(v2d2))
    cv2.putText(img2p, str(round(np.sum(all_matches == predicted) * 100.0 / len(predicted), 2)).format('.2f') + '%', \
        (500, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)



    vh_gt, topoh_gt = make_inter_graph(v2d1, v2d2, topo1, topo2, all_matches)
    vh_pred, topoh_pred = make_inter_graph(v2d1, v2d2, topo1, topo2, predicted)
    vh_gt_valid, topoh_gt_valid = make_inter_graph_valid(v2d1, v2d2, topo1, topo2, all_matches)
    vh_pred_valid, topoh_pred_valid = make_inter_graph_valid(v2d1, v2d2, topo1, topo2, predicted)
    v2d1t = ((v2d2[predicted] + v2d1) * 0.5).astype(int)
    v2d2t = ((v2d1[predicted1] + v2d2) * 0.5).astype(int)

    vh_gt = vh_gt.astype(int)
    vh_gt_valid = vh_gt_valid.astype(int)
    vh_pred = vh_pred.astype(int)
    vh_pred_valid = vh_pred_valid.astype(int)

    imgh = np.zeros_like(img1) + 255
    imghp = np.zeros_like(img1) + 255
    imgh_valid = np.zeros_like(img1) + 255
    imghp_valid = np.zeros_like(img1) + 255

    for node, nbs in enumerate(topoh_gt):
        for nb in nbs:
            cv2.line(imgh, [vh_gt[node][0], vh_gt[node][1]], [vh_gt[nb][0], vh_gt[nb][1]], [0, 0, 0], 2)
    
    for node, nbs in enumerate(topoh_pred):
        for nb in nbs:
            cv2.line(imghp, [vh_pred[node][0], vh_pred[node][1]], [vh_pred[nb][0], vh_pred[nb][1]], [0, 0, 0], 2)
    
    for node, nbs in enumerate(topoh_gt_valid):
        for nb in nbs:
            cv2.line(imgh_valid, [vh_gt_valid[node][0], vh_gt_valid[node][1]], [vh_gt_valid[nb][0], vh_gt_valid[nb][1]], [0, 0, 0], 2)
    
    for node, nbs in enumerate(topoh_pred_valid):
        for nb in nbs:
            cv2.line(imghp_valid, [vh_pred_valid[node][0], vh_pred_valid[node][1]], [vh_pred_valid[nb][0], vh_pred_valid[nb][1]], [0, 0, 0], 2)
    
    # for node, nbs in enumerate(topo1):
    #     for nb in nbs:
    #         cv2.line(imghp_valid, [v2d1t[node][0], v2d1t[node][1]], [v2d1t[nb][0], v2d1t[nb][1]], [0, 0, 0], 2)
    
    # for node, nbs in enumerate(topo2):
    #     for nb in nbs:
    #         cv2.line(imghp_valid, [v2d2t[node][0], v2d2t[node][1]], [v2d2t[nb][0], v2d2t[nb][1]], [0, 0, 0], 2)
    


    im_h = cv2.hconcat([img1, img2])
    im_hp = cv2.hconcat([img1p, img2p])
    img_inter = cv2.hconcat([imgh, imghp])
    img_inter_valid = cv2.hconcat([imgh_valid, imghp_valid])
    im_hv = cv2.vconcat([im_h, im_hp, img_inter, img_inter_valid])

    return im_hv
