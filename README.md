Practical Machine Learning Week 4 Project
Chuk Yong
17 June 2018

Barbell Lift Movement Analysis
Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).
Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3xsbS5bVX
The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
The data for this project comes from http://groupware.les.inf.puc-rio.br/har. Full source: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. “Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human ’13)”. Stuttgart, Germany: ACM SIGCHI, 2013.
Special thanks to the above mentioned authors for sharing their data.

Start of Data Analysis
1. Load and Clean Data
library(readr)
library(dplyr)
pml_training <- read_csv("D:/Coursera Data Science/Practical Machine Learning/week 4 project/pml-training.csv")
pml_testing <- read_csv("D:/Coursera Data Science/Practical Machine Learning/week 4 project/pml-testing.csv")
# Cleaning Data
NAcolumns <- colnames(pml_training[,which(colMeans(is.na(pml_training)) > 0.5)]) # Find columns with more than 50% NAs in them.
pml_training <- pml_training %>% select(-one_of(NAcolumns)) %>% filter(complete.cases(.)) # remove columns and rows with NA
pml_training <- select(pml_training,-(1:7)) # remove first 7 columns as they do not contain information needed for our analysis
# Do the same for testing data
pml_testing <- pml_testing %>% select(-one_of(NAcolumns)) %>% filter(complete.cases(.))
pml_testing <- select(pml_testing,-(1:7))
2. Learning on Training Data
library(caret)
set.seed(1688)
# Subset training data set so that we can train and test our predictors
inTrain <- createDataPartition(pml_training$classe, p=0.7, list = FALSE)
training <- pml_training[inTrain,]
testing <- pml_training[-inTrain,]

# Create a data frame to store overall accuracies for comparison
Accuracy <- data.frame(matrix(ncol = 3, nrow = 1))
colnames(Accuracy) <- c("DecisionTree", "RandomForest", "GBM")

# We will use 3 classification methods and compare their accuracy: Decision Tree, Random Forest and Generalized Boosted Regression Model (GBM)
# Start Decision Tree
library(rpart)
library(rpart.plot)
library(e1071)
modFitTree <- rpart(classe ~., data = training, method = "class")
predTree <- predict(modFitTree, newdata = testing, type = "class")
confTree <- confusionMatrix(predTree, as.factor(testing$classe))
Accuracy$DecisionTree <- as.numeric(confTree$overall['Accuracy'])
Decision Tree Plot
rpart.plot(modFitTree,tweak = 1.2, cex=0.3)

Accuracy of Decision Tree
confTree
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1522  174   23   58   13
##          B   52  662   91   88   91
##          C   51  161  828  152  119
##          D   19   76   56  608   54
##          E   29   66   28   58  805
## 
## Overall Statistics
##                                          
##                Accuracy : 0.752          
##                  95% CI : (0.7408, 0.763)
##     No Information Rate : 0.2843         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.6857         
##  Mcnemar's Test P-Value : < 2.2e-16      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9097   0.5812   0.8070   0.6307   0.7440
## Specificity            0.9364   0.9321   0.9006   0.9583   0.9623
## Pos Pred Value         0.8503   0.6728   0.6316   0.7478   0.8164
## Neg Pred Value         0.9631   0.9027   0.9567   0.9298   0.9434
## Prevalence             0.2843   0.1936   0.1744   0.1638   0.1839
## Detection Rate         0.2587   0.1125   0.1407   0.1033   0.1368
## Detection Prevalence   0.3042   0.1672   0.2228   0.1382   0.1676
## Balanced Accuracy      0.9231   0.7567   0.8538   0.7945   0.8531
Random Forest Analysis
modFitRF <- train(classe ~., data = training, method = "rf")
predRF <- predict(modFitRF, newdata = testing)
confRF <- confusionMatrix(predRF, as.factor(testing$classe))
Accuracy$RandomForest <- as.numeric(confRF$overall['Accuracy'])
Accuracy of Random Forest
confRF
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1671   11    0    0    0
##          B    2 1126    4    1    0
##          C    0    2 1019   21    2
##          D    0    0    3  941    1
##          E    0    0    0    1 1079
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9918         
##                  95% CI : (0.9892, 0.994)
##     No Information Rate : 0.2843         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9897         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9886   0.9932   0.9761   0.9972
## Specificity            0.9974   0.9985   0.9949   0.9992   0.9998
## Pos Pred Value         0.9935   0.9938   0.9761   0.9958   0.9991
## Neg Pred Value         0.9995   0.9973   0.9986   0.9953   0.9994
## Prevalence             0.2843   0.1936   0.1744   0.1638   0.1839
## Detection Rate         0.2840   0.1914   0.1732   0.1599   0.1834
## Detection Prevalence   0.2859   0.1926   0.1774   0.1606   0.1835
## Balanced Accuracy      0.9981   0.9936   0.9940   0.9877   0.9985
GBM Analysis
modFitGBM  <- train(classe ~ ., data=training, method = "gbm",verbose = FALSE)
predGBM <- predict(modFitGBM, newdata = testing)
confGBM <- confusionMatrix(predGBM, as.factor(testing$classe))
Accuracy$GBM <- as.numeric(confGBM$overall['Accuracy'])
Accuracy of GBM
confGBM
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1649   35    0    1    2
##          B   15 1075   23    3    6
##          C    5   26  986   37    9
##          D    2    2   17  917   12
##          E    2    1    0    6 1053
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9653          
##                  95% CI : (0.9603, 0.9699)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9561          
##  Mcnemar's Test P-Value : 9.493e-05       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9857   0.9438   0.9610   0.9512   0.9732
## Specificity            0.9910   0.9901   0.9841   0.9933   0.9981
## Pos Pred Value         0.9775   0.9581   0.9276   0.9653   0.9915
## Neg Pred Value         0.9943   0.9866   0.9917   0.9905   0.9940
## Prevalence             0.2843   0.1936   0.1744   0.1638   0.1839
## Detection Rate         0.2803   0.1827   0.1676   0.1558   0.1790
## Detection Prevalence   0.2867   0.1907   0.1807   0.1615   0.1805
## Balanced Accuracy      0.9883   0.9670   0.9726   0.9723   0.9857
Model Selection
How did the models compared?
Accuracy
##   DecisionTree RandomForest       GBM
## 1    0.7520394    0.9918423 0.9653297
As we can see, Random Forest has the highiest accuracy rate, followed closely by GBM. We shall use Random Forest as our predicto on the test data.
Prediction on Test Data Set
Here are the results of our prediction with the data from the testing set:
predFinal <- predict(modFitRF, newdata=pml_testing)
predFinal
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
