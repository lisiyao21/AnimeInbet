# AnimeInbet

Code for ICCV 2023 paper "Deep Geometrized Cartoon Line Inbetweening"

[[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Siyao_Deep_Geometrized_Cartoon_Line_Inbetweening_ICCV_2023_paper.pdf) | [[Video Demo]](https://youtu.be/iUF-LsqFKpI?si=9FViAZUyFdSfZzS5) | [[Data (Google Drive)]](https://drive.google.com/file/d/1SNRGajIECxNwRp6ZJ0IlY7AEl2mRm2DR/view?usp=sharing) 

✨ Do not hesitate to give a star! Thank you! ✨


![image](https://github.com/lisiyao21/AnimeInbet/blob/main/figures/inbet_gif.gif)

> We aim to address a significant but understudied problem in the anime industry, namely the inbetweening of cartoon line drawings. Inbetweening involves generating intermediate frames between two black-and-white line drawings and is a time-consuming and expensive process that can benefit from automation. However, existing frame interpolation methods that rely on matching and warping whole raster images are unsuitable for line inbetweening and often produce blurring artifacts that damage the intricate line structures. To preserve the precision and detail of the line drawings, we propose a new approach, AnimeInbet, which geometrizes raster line drawings into graphs of endpoints and reframes the inbetweening task as a graph fusion problem with vertex repositioning. Our method can effectively capture the sparsity and unique structure of line drawings while preserving the details during inbetweening. This is made possible via our novel modules, i.e., vertex geometric embedding, a vertex correspondence Transformer, an effective mechanism for vertex repositioning and a visibility predictor. To train our method, we introduce MixamoLine240, a new dataset of line drawings with ground truth vectorization and matching labels. Our experiments demonstrate that AnimeInbet synthesizes high-quality, clean, and complete intermediate line drawings, outperforming existing methods quantitatively and qualitatively, especially in cases with large motions.

# ML240 Data

The implementation of AnimeInbet depends on the matching of line vertcies in the two adjancent two frames. To supervise the learning of vertex correspondence, we make a large-scale cartoon line sequential data, **MixiamoLine240** (ML240). ML240 contains a training set (100 sequences), a validation set (44 sequences) and a test set (100 sequences). Each sequence i

To use the data, please first download it from [link](https://drive.google.com/file/d/1SNRGajIECxNwRp6ZJ0IlY7AEl2mRm2DR/view?usp=sharing) and uncompress it into **data** folder under this project directory. After decompression, the data will be like 

        data
          |_ml100_norm
          |        |_ all
          |             |_frames  
          |             |    |_chip_abe
          |             |    |     |_Image0001.png
          |             |    |     |_Image0001.png
          |             |    |     |
          |             |    |     ...  
          |             |    ... 
          |             |
          |             |_labels
          |                  |_chip_abe
          |                  |     |_Line0001.json
          |                  |     |_Line0001.json
          |                  |     |
          |                  |     ...  
          |                  ...
          | 
          |_ml144_norm_100_44_split  
                  |_ test
                  |    |_frames  
                  |    |    |_breakdance_1990_police
                  |    |    |     |_Image0001.png
                  |    |    |     |_Image0001.png
                  |    |    |     |
                  |    |    |     ...  
                  |    |    ... 
                  |    |
                  |    |_labels
                  |         |_breakdance_1990_police
                  |         |     |_Line0001.json
                  |         |     |_Line0001.json
                  |         |     |
                  |         |     ...  
                  |         ...
                  |_ train
                      |_frames  
                      |    |_breakdance_1990_ganfaul
                      |    |     |_Image0001.png
                      |    |     |_Image0001.png
                      |    |     |
                      |    |     ...  
                      |    ... 
                      |
                      |_labels
                          |_breakdance_1990_ganfaul
                          |     |_Line0001.json
                          |     |_Line0001.json
                          |     |
                          |     ...  
                          ...


The json file in the "labels" folder (for example, ml100_norm/all/labels/chip_abe/Line0001.json) is the verctorization/geometrization labels of the corresponding image in the "frames" folder (ml100_norm/ all/frames/chip_abe/_Image0001.png). Each json file contains there components. (1) **vertex location**: line art vertices 2D positions, (2) **connection**: adjancent table of the vector graph and (3) **original index**: the index number of each vertex in the original 3D mesh.


# Code

## Environment 

    * pytorch == 1.7


![image](https://github.com/lisiyao21/AnimeInbet/blob/main/figures/pipeline.png)

In this code, the whole pipeline is separated into two parts: (1) vertex correspondence and (2) inbetweening/synthesis. In the first part, it is trained to match the vertices of two input vector graphs, including the "vertex embedding" and "vertex corr. Transformer". Then,  "repositioning propagation" and "graph fusion" are done in the second part.

The first part is inner ./corr, and the second is all others. We provide a pretrained correspondence network weight ([link](https://drive.google.com/file/d/1Edc-XGyMXqXDdfBYoglDMkBf7_AYZU0p/view?usp=sharing)) and a pretrained whole pipeline weight ([link](https://drive.google.com/file/d/1cemJCBNdcTvJ9LWCA_5LmDDorwEb-u7M/view?usp=sharing)). For correspondence, please decompress the weight (epoch_50.pt) to ./corr/experiments/vtx_corr/ckpt. For the whole pipeline, please decompress the weight (epoch_20.pt) to ./experiments/inbetweener_full/ckpt/.


## Train & test corr.

For training, first, please cd into the ./corr folder and then run

    sh srun.sh configs/vtx_corr.yaml train [your node name] 1

If you don't use slurm in your computer/cluster, you can run

    python -u main.py --config vtx_corr.yaml --train 

For testing correspondence network, please run

    sh srun.sh configs/vtx_corr.yaml train [your node name] 1

or 

    python -u main.py --config vtx_corr.yaml --test

You may directly run the test code after downloading the weights without training.

## Train & test the whole inbetweening pipeline

For training the whole pipeline, please firstly cd out from ./corr to the root project folder and run

    sh srun.sh configs/cr_inbetweener_full.yaml train [your node name] 1

or

    python -u main.py --config cr_inbetweener_full.yaml --train 

For testing, please run

    sh srun.sh configs/cr_inbetweener_full.yaml train [your node name] 1

or 

    python -u main.py --config cr_inbetweener_full.yaml --test

Inbetweened results will be stored into ./inbetween_results folder.

### Compute CD values

The CD code is under utils/chamfer_distance.py. Please run

    python compute_cd.py --gt ./data/ml100_norm/all/frames --generated ./inbetween_results/test_gap=5

If everything goes right the score will be the same as that reported in the paper.


# Citation

If you use our code or data, or find our work inspiring, please kindly cite our paper:

    @inproceedings{siyao2023inbetween,
	    title={Deep Geometrized Cartoon Line Inbetweening,
	    author={Siyao, Li and Gu, Tianpei and Xiao, Weiye and Ding, Henghui and Liu, Ziwei and Loy, Chen Change},
	    booktitle={ICCV},
	    year={2023}
    }

# License

ML240 is released with CC BY-NC-SA 4.0. Code is released with MIT License.

