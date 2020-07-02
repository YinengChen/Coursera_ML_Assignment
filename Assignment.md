---
title: "assignment"
author: "Yineng Chen"
date: "6/30/2020"
output: 
  html_document:
    keep_md: True
editor_options: 
  chunk_output_type: console
---



## Task

Using data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants, create a model to predict how well a subject perform in weight lifting exercises.

### classification of performance

1. Exactly according to the specification (Class A)

2. Throwing the elbows to the front (Class B)

3. Lifting the dumbbell only halfway (Class C)

4. Lowering the dumbbell only halfway (Class D) 

5. Throwing the hips to the front (Class E)


## Data input

### Load Data




### Data Cleaning

According to the codebook, the data set includes summary statistics calculated at a measurement boundary. These statistics containes high proportions of missing values and/or error values. Therefore, these statistics are not used in training, instead, only raw data from accelerometers are used. The summary statistics have names begin with "kurtosis_", "skewness_", "max_", "min_", "amplitude_", "var_", "avg_", and "stddev_". Variables that are not related to prediction are also removed including X1. user_name, num_window,new_window,timestamp.


```r
# deal with missing values
kname = grep("^(kurtosis_|skewness_|max_|min_|amplitude_|var_|avg_|stddev_)|window|X1|name|time",names(trainset))
trainset = trainset[,-kname]
dim(trainset)
```

```
## [1] 19622    53
```

```r
names(trainset)
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
```


## Algorithm

### Slit the trainset into training set and testing set


```r
inTrain = createDataPartition(trainset$classe, p = 0.6, list = F)
training = trainset[inTrain,]
testing = trainset[-inTrain,]
dim(training)
```

```
## [1] 11776    53
```

```r
dim(testing)
```

```
## [1] 7846   53
```


### Creat a random forest model with cross validation




```r
myModelFilename <- "myModel.RData"
if (!file.exists(myModelFilename)){
  set.seed(95014)

  cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
  registerDoParallel(cluster)
  getDoParWorkers()


  fitControl <- trainControl(method = "cv",
                           number = 5,
                           allowParallel = TRUE,
                           p = 0.6)


  model1 <- train(classe~.,
                  data = training,
                  metric = "Accuracy",
                  method = "rf",
                preProcess = c("center","scale"),
                 trControl = fitControl)

   save(model1, file = "myModel.RData")
  
  stopCluster(cluster)
  registerDoSEQ()
  
  
} else {
    load(file = myModelFilename, verbose = TRUE)
}
```

```
## Loading objects:
##   model1
```


```r
print(model1, digits = 4)
```

```
## Random Forest 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered (52), scaled (52) 
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 9420, 9420, 9420, 9422, 9422 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa 
##    2    0.9885    0.9854
##   27    0.9895    0.9867
##   52    0.9778    0.9720
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```


## Predict


```r
pred <- predict(model1, newdata = testing)
length(pred)
```

```
## [1] 7846
```

```r
length(testing$classe)
```

```
## [1] 7846
```

## Evaluation


```r
confusionMatrix(pred, as.factor(testing$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2230    0    0    0    0
##          B    2 1514    1    1    0
##          C    0    4 1362    7    3
##          D    0    0    5 1277    2
##          E    0    0    0    1 1437
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9967          
##                  95% CI : (0.9951, 0.9978)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9958          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9974   0.9956   0.9930   0.9965
## Specificity            1.0000   0.9994   0.9978   0.9989   0.9998
## Pos Pred Value         1.0000   0.9974   0.9898   0.9945   0.9993
## Neg Pred Value         0.9996   0.9994   0.9991   0.9986   0.9992
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1930   0.1736   0.1628   0.1832
## Detection Prevalence   0.2842   0.1935   0.1754   0.1637   0.1833
## Balanced Accuracy      0.9996   0.9984   0.9967   0.9960   0.9982
```

## Final Model


```r
model1$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.83%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3343    4    1    0    0 0.001493429
## B   21 2251    6    1    0 0.012286090
## C    0   10 2036    8    0 0.008763389
## D    0    2   29 1896    3 0.017616580
## E    0    2    2    9 2152 0.006004619
```


## Quiz


```r
print(predict(model1, newdata=quizset))
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

