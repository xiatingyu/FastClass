# FastClass
 Code of paper "FastClass: A Time-Efficient Approach to Weakly-Supervised Text Classification"

We provide `run_sst.sh` to reproduce the results of FastClass, and take the **SST** dataset as an example. 

```python
bash run_sst.sh
```
You can also use other datasets for testing, compare with other datasets, SST has a smaller amount of data and a shorter running time.

We used `python=3.8`, `cudatoolkit=11.1`. Other packages can be installed via 
```python
pip install -r requirements.txt
```
Citation
----
If you use this code, please cite this paper:
```
@inproceedings{xia-etal-2022-fastclass,
    title = "{F}ast{C}lass: A Time-Efficient Approach to Weakly-Supervised Text Classification",
    author = "Xia, Tingyu  and Wang, Yue  and Tian, Yuan  and Chang, Yi",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    year = "2022",
    pages = "4746--4758",
}

```
