# Out-of-distribution Detection with Deep Nearest Neighbors (Modified)

This repository is based on the original implementation of the ICML 2022 paper [Out-of-distribution Detection with Deep Nearest Neighbors](https://arxiv.org/abs/2204.06507)
by Yiyou Sun, Yifei Ming, Xiaojin Zhu and Yixuan Li.

We have modified and automated the original codebase for experiments using CIFAR-10 as the in-distribution dataset, and iNaturalist and SUN as out-of-distribution (OOD) datasets.

## Google Colab Notebook

All steps including dataset download, preprocessing, model loading, and OOD evaluation are fully automated and executed in Google Colab.
The notebook mounts Google Drive for persistent dataset and model storage.

🔗 Colab Notebook: [Run on Colab](https://colab.research.google.com/drive/1FdD6jyVyAKU5coWtsZVCo6DD0yXAMxSn#scrollTo=rYZITUJGECiv)


## Citation


If you use our codebase, please cite our work:

```
@article{sun2022knnood,
  title={Out-of-distribution Detection with Deep Nearest Neighbors},
  author={Sun, Yiyou and Ming, Yifei and Zhu, Xiaojin and Li, Yixuan},
  journal={ICML},
  year={2022}
}
```

