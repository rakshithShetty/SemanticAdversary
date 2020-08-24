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
# Setup
## Tested on
* Python 3.6.7
* PyTorch 1.5

Download the coco dataset into data/coco directory
Download the json file below which collects all the required meta data for COCO into the data/coco directory
Download the pretrained models given below.

To run interpolations between randomly sampled objects and compare results run the script as below.
```
CUDA_VISIBLE_DEVICES=X python compare_interpolations.py -m modelA.pth.tar modelB.pth.tar  -n savename_modelA savename_modelB  
```
Code to train the synthesizer network and to run adversarial attack will be released soon.

# Downloads
* [Pre-trained synthesizer model (our part-segmentation bottlneck)](https://datasets.d2.mpi-inf.mpg.de/rakshith/semadv/ourpas18model.pth.tar) - Trained on COCO dataset for matching 18 pascal object categories. 
* [Alternate synthesizer model (Gaussian bottlneck)](https://datasets.d2.mpi-inf.mpg.de/rakshith/semadv/gausspas18model.pth.tar) - Trained on COCO dataset for matching 18 pascal object categories. 
* [COCO dataset file](https://datasets.d2.mpi-inf.mpg.de/rakshith/semadv/datasetCompleteYoloSplit.json) - Single json file with metadata and annotations for the COCO dataset

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
