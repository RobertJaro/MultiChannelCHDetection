![](images/title.jpg)
---
# Multi-channel coronal hole detection with convolutional neural networks

# [Paper](#paper) --- [Code and Data](#code) --- [Citation](#citation) --- [Contact](#contact)

## Abstract

_Context._ A precise detection of the coronal hole boundary is of primary interest for a better understanding of the physics of coronal
holes, their role in the solar cycle evolution, and space weather forecasting.

_Aims._ We develop a reliable, fully automatic method for the detection of coronal holes that provides consistent full-disk segmentation
maps over the full solar cycle and can perform in real-time.

_Methods._ We use a convolutional neural network to identify the boundaries of coronal holes from the seven extreme ultraviolet (EUV)
channels of the Atmospheric Imaging Assembly (AIA) and from the line-of-sight magnetograms provided by the Helioseismic and
Magnetic Imager (HMI) on board the Solar Dynamics Observatory (SDO). For our primary model (Coronal Hole RecOgnition Neural
Network Over multi-Spectral-data; CHRONNOS) we use a progressively growing network approach that allows for efficient training,
provides detailed segmentation maps, and takes into account relations across the full solar disk.

_Results._ We provide a thorough evaluation for performance, reliability, and consistency by comparing the model results to an independent
manually curated test set. Our model shows good agreement to the manual labels with an intersection-over-union (IoU) of
0.63. From the total of 261 coronal holes with an area > 1.5e10 km2 identified during the time-period from November 2010 to December
2016, 98.1% were correctly detected by our model. The evaluation over almost the full solar cycle no. 24 shows that our model
provides reliable coronal hole detections independent of the level of solar activity. From a direct comparison over short timescales of
days to weeks, we find that our model exceeds human performance in terms of consistency and reliability. In addition, we train our
model to identify coronal holes from each channel separately and show that the neural network provides the best performance with
the combined channel information, but that coronal hole segmentation maps can also be obtained from line-of-sight magnetograms
alone.

_Conclusions._ The proposed neural network provides a reliable data set for the study of solar-cycle dependencies and coronal-hole
parameters. Given the fast and robust coronal hole segmentation, the algorithm is also highly suitable for real-time space weather
applications.

---

## Code

The code and data set will be made publicly available with the next CHRONNOS version, covering data from SDO, SOHO and STEREO.
This study will also feature a first statistical evaluation of coronal holes since 1996 at high cadence.

## Paper

Journal Version (A&A): https://doi.org/10.1051/0004-6361/202140640

Open access (arXiv): https://arxiv.org/abs/2104.14313

## Citation


```
@article{Jarolim2021chronnos,
    title={Multi-channel coronal hole detection with convolutional neural networks},
    url={http://dx.doi.org/10.1051/0004-6361/202140640}, 
    DOI={10.1051/0004-6361/202140640}, 
    journal={Astronomy & Astrophysics}, 
    author={Jarolim, R. and Veronig, A. M. and Hofmeister, S. and Heinemann, S. G. and Temmer, M. and Podladchikova, T. and Dissauer, K.}, 
    year={2021}, 
    month={May}
}
```

## Contact

Robert Jarolim<br/>
[robert.jarolim@uni-graz.at](mailto:robert.jarolim@uni-graz.at)

![](images/samples.jpg)
