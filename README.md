# UM2Former: U-Shaped Multimixed Transformer Network for Large-Scale Hyperspectral Image Semantic Segmentation

[Aijun Xu](https://www.researchgate.net/profile/Xu-Aijun?ev=hdr_xprf), [Zhaohui Xue](https://sites.google.com/site/zhaohuixuers), [Ziyu Li](https://www.researchgate.net/profile/Ziyu-Li-28), Shun Cheng, [Hongjun Su](https://www.researchgate.net/profile/Hongjun-Su), [Junshi Xia](https://www.researchgate.net/profile/Junshi-Xia)

---

The code in this toolbox implements the ["UM2Former: U-Shaped Multimixed Transformer Network for Large-Scale Hyperspectral Image Semantic Segmentation"](https://ieeexplore.ieee.org/document/10892222), which has been published at *IEEE Transactions on Geoscience and Remote Sensing* !  

![Figure 1.](/UM2Former/figure/UM2Former.png)

Figure 1. The overview of UM2Former, including four stages of weighted encoder, multimixed Transformer(MMTB) and a linear-fuse segmentation head(LFSH).

---

![Figure 2.](/UM2Former/figure/weighted_encoder.png)

Figure 2. Graphical illustration of the proposed weighted encoder, including overlap-down and channel-weight.

---

![Figure 3.](/UM2Former/figure/MMTB.png)

Figure 3. Graphical illustration of the proposed MMTB, including spatial-feature-retention attention mechanism, which is a positional encoding free module.

---

![Figure 4.](/UM2Former/figure/LFSH.png)

Figure 4. Graphical illustration of the proposed LFSH.  

## Dataset

The large spatial coverage and extensive classification scene make the [WHU-OHS](https://github.com/zjjerica/WHU-OHS-Pytorch) dataset a challenging benchmark for large-scale HSI semantic segmentation.

## Training & Testing

**Reproductions of our model and experiments are very welcome!**

❗ Please refer to the `Requirements` for the running environments of this code.

❗❗ The trained model parameters can be used in scripts `test/test_UM2Former.py` and `predict/predict_UM2Former.py` to replicate the accuracy and prediction results in our paper.

❗❗❗ The trained model parameters for WHU-OHS dataset can be downloaded from the following links:

[Baiduyun](https://pan.baidu.com/s/13O8dk1mcyMQlE5MvgjDWUQ) (access code: rscv)

## Citation

**Citations to our paper will be greatly appreciated!**  

A. Xu *et al*, "UM2Former: U-Shaped Multimixed Transformer Network for Large-Scale Hyperspectral Image  Semantic Segmentation," in *IEEE Transactions on Geoscience and Remote Sensing*, vol. 63, pp. 1-21, 2025, Art no. 5506221, doi: 10.1109/TGRS.2025.3543821.

```
@ARTICLE{10892222,
  author={Xu, Aijun and Xue, Zhaohui and Li, Ziyu and Cheng, Shun and Su, Hongjun and Xia, Junshi},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={UM2Former: U-Shaped Multimixed Transformer Network for Large-Scale Hyperspectral Image Semantic Segmentation}, 
  year={2025},
  volume={63},
  number={},
  pages={1-21},
  doi={10.1109/TGRS.2025.3543821}
}
```

## Requirements

Running environment and required packages:

```
timm==1.0.9
tqdm==4.66.1
numpy==1.23.5
einops==0.7.0
ptflops==0.6.9
python==3.9.19
matplotlib==3.7.2
torch==1.12.0+cu113
torch-summarry==1.4.5
mmsegmentation==1.2.2
```

## Instructions for usage

```
Nets ..... A file for semantic segmentation models, including the implementation of UM2Former.
Unet ..... A file for the implementation of UNet.
UnetFormer ..... A file for the implementation of UNetFormer.
train ..... A file for training UM2former, including trained model parameter.
test ..... A file for testing evaluation metrics of UM2Former.
predict ..... A file for showing prediction results of UM2Former.
```
