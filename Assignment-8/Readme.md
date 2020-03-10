# Assignment - 8

- Epochs: 25
- Max_Val_Acc: 90.50% (ep=23)
- Final_Val_Acc: 90.18 % (ep=24)
- Augmentation Strategy: Random Horizontal Flip on Train data, Normalization on test and train
- Resnet18
- All the py files are called from Main.ipynb files in a Modular way

## Accuracies

Accuracy of plane : 93 %
Accuracy of   car : 100 %
Accuracy of  bird : 84 %
Accuracy of   cat : 73 %
Accuracy of  deer : 82 %
Accuracy of   dog : 83 %
Accuracy of  frog : 88 %
Accuracy of horse : 96 %
Accuracy of  ship : 95 %
Accuracy of truck : 94 %

# Graph

![Acc and Loss](https://github.com/divyanshbajpai/EVA4-Coursework/blob/master/Assignment-8/s8.png)

# Logs

 0%|          | 0/391 [00:00<?, ?it/s]EPOCH: 0
Loss=1.277374 Batch_id=390 Accuracy=53.36: 100%|██████████| 391/391 [00:29<00:00, 14.25it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.996985, Accuracy: 6495/10000 (64.95%)

EPOCH: 1
Loss=0.771667 Batch_id=390 Accuracy=73.01: 100%|██████████| 391/391 [00:29<00:00, 14.38it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.743407, Accuracy: 7412/10000 (74.12%)

EPOCH: 2
Loss=0.601596 Batch_id=390 Accuracy=79.05: 100%|██████████| 391/391 [00:29<00:00, 14.46it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.893401, Accuracy: 6853/10000 (68.53%)

EPOCH: 3
Loss=0.509866 Batch_id=390 Accuracy=82.51: 100%|██████████| 391/391 [00:29<00:00, 14.28it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.609205, Accuracy: 7886/10000 (78.86%)

EPOCH: 4
Loss=0.450067 Batch_id=390 Accuracy=84.58: 100%|██████████| 391/391 [00:29<00:00, 14.24it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.613217, Accuracy: 7889/10000 (78.89%)

EPOCH: 5
Loss=0.411989 Batch_id=390 Accuracy=85.94: 100%|██████████| 391/391 [00:29<00:00, 14.51it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.651280, Accuracy: 7752/10000 (77.52%)

EPOCH: 6
Loss=0.380549 Batch_id=390 Accuracy=87.16: 100%|██████████| 391/391 [00:29<00:00, 14.26it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.717997, Accuracy: 7655/10000 (76.55%)

EPOCH: 7
Loss=0.357670 Batch_id=390 Accuracy=87.89: 100%|██████████| 391/391 [00:29<00:00, 14.26it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.582690, Accuracy: 7981/10000 (79.81%)

EPOCH: 8
Loss=0.213420 Batch_id=390 Accuracy=93.41: 100%|██████████| 391/391 [00:29<00:00, 14.26it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.313500, Accuracy: 8928/10000 (89.28%)

EPOCH: 9
Loss=0.157498 Batch_id=390 Accuracy=95.54: 100%|██████████| 391/391 [00:29<00:00, 14.29it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.295982, Accuracy: 8992/10000 (89.92%)

EPOCH: 10
Loss=0.130601 Batch_id=390 Accuracy=96.56: 100%|██████████| 391/391 [00:29<00:00, 14.34it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.290358, Accuracy: 8977/10000 (89.77%)

EPOCH: 11
Loss=0.108410 Batch_id=390 Accuracy=97.25: 100%|██████████| 391/391 [00:29<00:00, 14.42it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.301569, Accuracy: 8986/10000 (89.86%)

EPOCH: 12
Loss=0.089261 Batch_id=390 Accuracy=98.00: 100%|██████████| 391/391 [00:29<00:00, 14.51it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.297484, Accuracy: 8967/10000 (89.67%)

EPOCH: 13
Loss=0.071991 Batch_id=390 Accuracy=98.54: 100%|██████████| 391/391 [00:29<00:00, 14.55it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.294277, Accuracy: 8991/10000 (89.91%)

EPOCH: 14
Loss=0.056569 Batch_id=390 Accuracy=99.10: 100%|██████████| 391/391 [00:29<00:00, 14.34it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.303584, Accuracy: 8994/10000 (89.94%)

EPOCH: 15
Loss=0.048435 Batch_id=390 Accuracy=99.19: 100%|██████████| 391/391 [00:28<00:00, 14.67it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.299693, Accuracy: 9030/10000 (90.30%)

EPOCH: 16
Loss=0.033986 Batch_id=390 Accuracy=99.66: 100%|██████████| 391/391 [00:29<00:00, 14.36it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.293128, Accuracy: 9016/10000 (90.16%)

EPOCH: 17
Loss=0.030151 Batch_id=390 Accuracy=99.77: 100%|██████████| 391/391 [00:29<00:00, 14.67it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.286741, Accuracy: 9022/10000 (90.22%)

EPOCH: 18
Loss=0.029400 Batch_id=390 Accuracy=99.77: 100%|██████████| 391/391 [00:29<00:00, 14.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.296543, Accuracy: 8991/10000 (89.91%)

EPOCH: 19
Loss=0.028063 Batch_id=390 Accuracy=99.80: 100%|██████████| 391/391 [00:29<00:00, 14.26it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.294173, Accuracy: 9032/10000 (90.32%)

EPOCH: 20
Loss=0.027422 Batch_id=390 Accuracy=99.81: 100%|██████████| 391/391 [00:29<00:00, 14.44it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.293876, Accuracy: 9037/10000 (90.37%)

EPOCH: 21
Loss=0.026057 Batch_id=390 Accuracy=99.84: 100%|██████████| 391/391 [00:29<00:00, 14.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.292441, Accuracy: 9028/10000 (90.28%)

EPOCH: 22
Loss=0.024898 Batch_id=390 Accuracy=99.88: 100%|██████████| 391/391 [00:29<00:00, 14.16it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.292945, Accuracy: 9018/10000 (90.18%)

EPOCH: 23
Loss=0.024979 Batch_id=390 Accuracy=99.88: 100%|██████████| 391/391 [00:30<00:00, 14.35it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.291242, Accuracy: 9050/10000 (90.50%)

EPOCH: 24
Loss=0.024182 Batch_id=390 Accuracy=99.88: 100%|██████████| 391/391 [00:29<00:00, 14.37it/s]

Test set: Average loss: 0.293802, Accuracy: 9043/10000 (90.43%)
