# LrAdjustmentMechanism-Pytorch-master

There are seven types of Learning rate adjustment scheduler mechanisms by commonly used!

It's just a demo that observe the learning rate, so maybe it doesn't work well! 

**AlexNet** was taken as the network architecture(**image size** required for input was **227x227x3**), **CIFAR10** as the datset, **SGD** as the gradient descent function.

You should create your own data(dataset), checkpoint(model save) and tenshorboard(loss visualization) file package.


and there are some my Chinese communication websites such as CSDN and Quora(Chinese)-Zhihu where I explain this code.

[CSDN](https://blog.csdn.net/XiaoyYidiaodiao/article/details/124678206?spm=1001.2014.3001.5501)

[Quora(Chinese)](https://zhuanlan.zhihu.com/p/511964712)

---

## 1.adjust_learning_rate()

Function description: subsection, every several (2) epochs, the first epoch is the serial number 0, so that the learning rate change multiplied by 0.1 epoch power number

run train function: `python train_adjust_learning_rate.py`

|     epoch   |  0-1  |  2-3  |  4-5  |   6-7 |   8-9   |   10-11   |  12-13  |  14-15 |  16-17  | 18-19 |
|-------------|-------|-------|-------|-------|---------|-----------|---------|--------|---------|-------|
|learning rate| 0.001 | 0.0001| 1e-05 | 1e-06 |  1e-07  |   1e-08   |  1e-09  |  1e-10 |  1e-11  | 1e-12 |


run eval function: `python eval.py`

|  AP |  14.82  |
|-----|---------|

---

## 2.MultiStepLR()

`milestones=[5, 10, 15], gamma=0.1`

run train function: `python train_MultiStepLR.py`

|     epoch   |  0-4  |  5-9  |  10-14  |  15-19 | 
|-------------|-------|-------|---------|--------|
|learning rate| 0.001 | 0.0001|  1e-05  |  1e-06 |

run eval function: `python eval.py`

|  AP |  18.16  |
|-----|---------|

---

## 3.StepLR()

`step_size=5, gamma=0.2`

run train function: `python train_StepLR.py`

|     epoch   |  0-4  |  5-9  |  10-14  |  15-19 | 
|-------------|-------|-------|---------|--------|
|learning rate| 0.001 | 0.0002|  4e-05  |  8e-06 |

run eval function: `python eval.py`

|  AP |  17.67  |
|-----|---------|

---

## 4.LambdaLR() 

`lambda1 = lambda epoch: (epoch) // 2`

run train function: `python train_LambdaLR.py`

|     epoch   |  0-1  |  2-3  |  4-5  |   6-7 |   8-9   |   10-11   |  12-13  |  14-15 |  16-17  | 18-19 |
|-------------|-------|-------|-------|-------|---------|-----------|---------|--------|---------|-------|
|learning rate|   0   | 0.001 | 0.002 | 0.003 |  0.004  |   0.005   |  0.006  |  0.007 |  0.008  | 0.009 |

run eval function: `python eval.py`

|  AP |  51.27  |
|-----|---------|

---

## 5.ReduceLROnPlateau()

run train function: `python train_ReduceLROnPlateau.py`

|     epoch   |   0-9  | 10-15  | 16-19  |
|-------------|--------|--------|--------|
|learning rate| 0.001  | 0.0001 |  1e-05 |

run eval function: `python eval.py`

|  AP |  18.00  |
|-----|---------|

---

## 6.ExponentialLR()

run train function: `python train_ExponentialLR.py`


|     epoch   |   0   |   1    |   2   |   3   |     4     |     5     |     6    |     7    |      8     |     9    |
|-------------|-------|--------|-------|-------|-----------|-----------|----------|----------|------------|----------|
|learning rate| 0.001 | 0.0002 | 4e-05 | 8e-06 |  1.6e-06  |  3.2e-07  |  6.4e-08 | 1.28e-08 |  2.56e-09  | 5.12e-10 |


|     epoch   |     10    |      11     |     12    |     13    |      14      | 
|-------------|-----------|-------------|-----------|-----------|--------------|
|learning rate| 1.024e-10 |  2.048e-11  | 4.096e-12 | 8.192e-13 |  1.6384e-13  |


 |     epoch   |       15     |     16      |      17     |       18      |      19     |
 |-------------|--------------|-------------|-------------|---------------|-------------|
 |learning rate|  3.2768e-14  |  6.5536e-15 | 1.31072e-15 |  2.62144e-16  | 5.24288e-17 |


run eval function: `python eval.py`

|  AP |  10.00  |
|-----|---------|

---

## 7.CosineAnnealingLR()

`T_max=5`

run train function: `python train_ExponentialLR.py`


|     epoch   |   0   |           1           |           2           |             3          |         4             | 
|-------------|-------|-----------------------|-----------------------|------------------------|-----------------------|
|learning rate| 0.001 | 0.0009045084971874737 | 0.0006545084971874737 | 0.00034549150281252633 | 9.549150281252633e-05 |




|     epoch   |     5     |           6           |            7           |           8           |            9          |
|-------------|-----------|-----------------------|------------------------|-----------------------|-----------------------|
|learning rate|   0.0     | 9.549150281252627e-05 | 0.00034549150281252617 | 0.0006545084971874735 | 0.0009045084971874735 |


|     epoch   |     10                  |           11           |            12           |           13           |            14          |
|-------------|-------------------------|------------------------|-------------------------|------------------------|------------------------|
|learning rate|  0.0009999999999999998  | 0.0009045084971874739  | 0.0006545084971874736   | 0.00034549150281252633 | 9.549150281252635e-05  |


|     epoch   |   15   |           16           |           17           |           18           |            19           |
|-------------|--------|------------------------|------------------------|------------------------|-------------------------|
|learning rate|   0.0  | 9.549150281252627e-05  | 0.00034549150281252617 |  0.0006545084971874743 |  0.0009045084971874746  |

run eval function: `python eval.py`

|  AP |  19.47  |
|-----|---------|

---



