# Ensemble Regularized Adaptive Prediction Set ERAPS
> Implementation and experiments based on the paper [Conformal prediction set for time-series](). The current paper is strongly accepted by the Workshop on Distribution-Free Uncertainty Quantification at ICML 2022

> Please direct all implementation-related inquiries to Chen Xu @ cxu310@gatech.edu.

> Citation:
```
  @misc{xuERPAS2022,
  doi = {10.48550/ARXIV.2206.07851},
  url = {https://arxiv.org/abs/2206.07851},
  author = {Xu, Chen and Xie, Yao},
  keywords = {Machine Learning (stat.ML), Machine Learning (cs.LG), Methodology (stat.ME), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Conformal prediction set for time-series},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}

```
## Documentation
- Executing the file [ERAPS_test.py](https://github.com/hamrel-cxu/Ensemble-Regularized-Adaptive-Prediction-Set-ERAPS/blob/main/ERAPS_test.py) reproduces our results. Detailed documentation will be provided soon.
- The Table below compares `ERAPS` vs. three competing methods on the [MelbournePedestrian dataset](https://www.timeseriesclassification.com/description.php?Dataset=MelbournePedestrian), where we can see that ERAPS always maintains valid marginala coverage with much smaller prediction sets in some cases.

Comparison of `ERAPS` vs. competitors         |
:-------------------------:
![](https://github.com/hamrel-cxu/Ensemble-Regularized-Adaptive-Prediction-Set-ERAPS/blob/main/Illustrative_fig.png)
