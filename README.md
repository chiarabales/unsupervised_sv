## Unsupervised Features Ranking via Coalitional Game Theory for Categorical Data

Welcome to the code for our paper, *Unsupervised Features Ranking via Coalitional Game Theory for Categorical Data*, published at **DaWaK 2022**. We encourage you to read the [full paper](https://arxiv.org/abs/2205.09060).

### Citation
If you found this work useful, please cite our paper:

```
@inproceedings{balestraUSV,
  author    = {Chiara Balestra  and
               Florian Huber and
               Andreas Mayr and
               Emmanuel M\"uller},
  title     = {Unsupervised Features Ranking via Coalitional Game Theory for Categorical Data},
  booktitle = {DaWaK 2022},
  publisher = {Springer},
  year      = {2022}
 }
```

### Example and code
The `shapley_calculation.py` contained the Shapley values implementation and `feature_selection.py` includes the SVFR and SVFS algorithms.
We provide an example of the use of the code in `_example.py` referring to a synthetic data set.

In order to run the code use
>
> `python example.py --_algorithm='SVFR' --_type='full' --_epsilon=0.6 --_approx=3 --_subsets_bound=2`
> 
and change the parameters as desired.

### Requirements
Code tested under:
- python 3.7.6
- numpy 1.18.5
- pandas 1.4.0

External librarys used:
- pyitlib 0.2.2 (pip install pyitlib)

### Questions
You can reach out to chiara.balestra1@gmail.com with any question

