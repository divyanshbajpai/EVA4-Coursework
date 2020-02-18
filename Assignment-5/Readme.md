# Code-1

- Target: Getting basic skeleton ready
- Result: The model works but number of parameter is huge
  - Parameters: 6,379,786
  - Best Train Accuracy: 99.95%
  - Best Test Accuracy: 99.25%
- Analysis:
 - The model performs well and rarely reached accuracy of 99.3%
 - The number of parameters are huge penalty.
 - Model needs to refine and cut down lot of parameters
 - The model is being overfitting, leaving no scope for further learning.

# Code-2

- Target: Decreasing the number of parameters.
- Result: The number of parameters decreased drastically
  - Parameters: 9,752
  - Best Train Accuracy: 99.66%
  - Best Test Accuracy: 99.32%
- Analysis:
 - We used squeeze and expand architecture. Using minimal number of channels required.
 - Using 1*1 frequently made it possible to control parameter graowth
 - Still model is overfitting and the 99.3 accuracy is rare, with huge difference between val acc and train acc.
 - Scope for more improvement

# Code-3

- Target: Tackle overfitting by adding random dropouts on train images
- Result: The model is now not overfitting.
  - Parameters: 9,752
  - Best Train Accuracy: 99.14%
  - Best Test Accuracy: 99.39%
- Analysis:
 - Adding Dropout increased the capacity of the model to learn more.
 - The difference between val acc and train acc is now very less.
 - Though no major difference in val acc but the model is now capable to learn more.

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

