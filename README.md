# Siam-NestedUNet 

## Introduction

![image.png](https://i.loli.net/2021/11/26/kuJWxPLg4EzmCVq.png)

Network for Change Detection. The pytorch implementation for "[SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images](https://ieeexplore.ieee.org/document/9355573) ". For details, please see [blog](https://blog.csdn.net/weixin_42392454/article/details/121557001?spm=1001.2014.3001.5501).
If this works for you, please give me a star, this is very important to me.😊
## Quick start

1. clone this repository

```shell
git clone https://github.com/Runist/Siam-NestedUNet
```
2. You need to install some dependency package.

```shell
cd Siam-NestedUNet
pip install -r requirements.txt
```
3. Download the change detection dataset. You can find the dataset  in this [repository](https://github.com/likyoo/Siam-NestedUNet#dataset).
4. Configure the parameters in [utils/parser.py](https://github.com/Runist/Siam-NestedUNet/blob/master/utils/parser.py).
5. Start train your model.

```shell
python train.py
```
6. Open tensorboard to watch training process.

```shell
tensorboard --logdir ./weights/tutorial/log/
```

![image.png](https://i.loli.net/2021/11/26/34wRqELWZMHCdal.png)

7. You can run *evaluate.py* to watch model performance.

```shell
python evaluate.py
```
8. Get prediction of model.

```shell
python inference.py
```

## Reference

Appreciate the work from the following repositories:

- [likyoo](https://github.com/likyoo)/[Siam-NestedUNet](https://github.com/likyoo/Siam-NestedUNet)

## License

Code and datasets are released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.
