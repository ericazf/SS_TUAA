# SS-TUAA
Official implementation of the ICASSP2024 paper: Exploring Targeted Universal Adversarial Attack for Deep Hashing

## Train

#### Deep hashing models 
```
cd Hash
python DPSH.py
python DSDH.py
```

#### TUAP generation
We first generate anchor code by running
```
cd attacks
python CAE.py
```
Under the supervision of CAE, and we can generate TUAP by running
```
cd attacks
python UAP.py
```



## Citation

If you find our code useful, please consider citing our paper:

```
@inproceedings{zhu2024exploring,
  title={Exploring Targeted Universal Adversarial Attack for Deep Hashing},
  author={Zhu, Fei and Zhang, Wanqian and Wu, Dayan and Wang, Lin and Li, Bo and Wang, Weiping},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={3335--3339},
  year={2024},
  organization={IEEE}
}
```

