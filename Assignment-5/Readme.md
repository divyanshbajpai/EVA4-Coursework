# Code-4

- Target: Image Augementation, Rotation
- Result: Training Accuracy decreased
  - Parameters: 9,752
  - Best Train Accuracy: 99.02%
  - Best Test Accuracy: 99.36%
- Analysis:
 - After adding rotation, the model is underfitting, the training accuracy has descreased.
 - This happens beacause we have made changes to train dataset and model will be learning new features(slightly randomly rotated images)

# Code-5
- Target: Change Learning Rate
- Result: The val accuracy increased. 99.4%+ acc is more often.
  - Parameters: 9,752
  - Best Train Accuracy: 99.20%
  - Best Test Accuracy: 99.44%
- Analysis:
 - The model is performing well, as more and more 99.4% above val acc is seen
 - Many hits and trial were required, as little change to LR changed the accuracies drastically.
 - Still not sure wheather this LR is the best? Many more permuatation and combinations are required.

