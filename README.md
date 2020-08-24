# Semantic Adversary
Code for our paper "Towards Automated Testing and Robustification by Semantic Adversarial Data Generation"
* [](https://datasets.d2.mpi-inf.mpg.de/rakshith/object_removal_nips/NIPS2018_poster.pdf)
![Teaser](/images/Teaser.png)
# Explanatory Video
[![Video](https://img.youtube.com/vi/1TiXRTJJikE/0.jpg)](https://www.youtube.com/watch?v=1TiXRTJJikE)


<!---
# Interpolation results
![Teaser](/gen_samples/gauss_vs_ours/train_COCO_train2014_000000184654.png)
-->
# Downloads
* [Pre-trained synthesizer model (our part-segmentation bottlneck)]() - Trained on COCO dataset for matching 18 pascal object categories. 
* [Alternate synthesizer model (Gaussian bottlneck)]() - Trained on COCO dataset for matching 18 pascal object categories. 
* [COCO dataset file](https://datasets.d2.mpi-inf.mpg.de/rakshith/object_removal_nips/datasetBoxAnn_80pcMaxObj_mrcnnval.json) - Single json file with metadata and annotations for the COCO dataset

# Bibtex
If you find this code useful in your work, please cite the paper.
```
PaperBibtex
@inproceedings{shetty2020SemAdv,
title={Towards automated testing and robustification by semantic adversarial data generation},
author={Shetty, Rakshith and Fritz, Mario and Schiele, Bernt},
booktitle={European Conference on Computer Vision (ECCV)},
year={2020},
}
```
