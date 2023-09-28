import argparse
import cv2
import os
from utils.chamfer_distance import cd_score
import numpy as np




if __name__ == "__main__":
    cds = []

    parser = argparse.ArgumentParser()
    parser.add_argument('--generated', type=str)
    parser.add_argument('--gt', type=str)
    args = parser.parse_args()

    gen_dir = args.generated
    gt_dir = args.gt

    if True:
    
        print('computing CD...', flush=True)

        for subfolder in os.listdir(gt_dir):
            # print(subfolder, len(cds), flush=True)
            for img in os.listdir(os.path.join(gt_dir, subfolder)):
                if not img.endswith('.png'):
                    continue
                img_gt = cv2.imread(os.path.join(gt_dir, subfolder, img))

                pred_name = subfolder + '_' + img.replace('Image', 'Line')
                if not os.path.exists(os.path.join(gen_dir, pred_name)):
                    continue
                img_pred = cv2.imread(os.path.join(gen_dir, pred_name))

                this_cd = cd_score(img_gt, img_pred)
                cds.append(this_cd)
                # print(this_cd, flush=True)
        
        print('GT: ', gt_dir)
        print('>>> Gen: ', gen_dir)
        print('>>> CD: ', np.mean(cds)/1e-5, print(len(cds)))



