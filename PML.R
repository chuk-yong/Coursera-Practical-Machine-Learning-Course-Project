library(readr)
library(dplyr)
pml_training <- read_csv("D:/Coursera Data Science/Practical Machine Learning/week 4 project/pml-training.csv")
pml_testing <- read_csv("D:/Coursera Data Science/Practical Machine Learning/week 4 project/pml-testing.csv")
# pml_training1 <- pml_training[, -which(colMeans(is.na(pml_training)) > 0.5)] # remove columns with more than 50% NA
# pml_training2 <- pml_training %>% select(-one_of(NAcolumns))
# pml_training3 <- pml_training[,colSums(is.na(pml_training)) ==0]
# NAcolumns <- colnames(pml_training)[colSums(is.na(pml_training)) > 0]
#pml_training1 has 60 columns while pml_training2 and 3 has 57.  Checking what is missing
# colTrain1 <- colnames(pml_training1)
# colTrain2 <- colnames(pml_training2)
# colTrain3 <- colnames(pml_training3)
# setdiff(colTrain1, colTrain2)
# "magnet_dumbbell_z" "magnet_forearm_y"  "magnet_forearm_z" .  no NA in them, why was it removed using the other two method?

# Cleaning Data
NAcolumns <- colnames(pml_training[,which(colMeans(is.na(pml_training)) > 0.5)]) # Find columns with more than 50% NAs in them.
pml_training <- pml_training %>% select(-one_of(NAcolumns)) # remove columns with NA
pml_training <- select(pml_training,-(1:7)) # remove first 7 columns as they do not contain information needed for our analysis
# Do the same for testing data
pml_testing <- pml_testing %>% select(-one_of(NAcolumns))
pml_testing <- select(pml_testing,-(1:7))



library(caret)
set.seed(1688)
# Subset training data set so that we can train and test our predictors
inTrain <- createDataPartition(pml_training$classe, p=0.7, list = FALSE)
training <- pml_training[inTrain,]
testing <- pml_training[-inTrain,]

library(rpart)
library(rpart.plot)
library(e1071)
modFitTree <- rpart(classe ~., data = training, method = "class")
rpart.plot(modFitTree,tweak = 1.2, cex=0.3)
# library(RColorBrewer)
# library(rattle)
# fancyRpartPlot(modFitTree, cex=0.3)
# plot is cluttered.  tweak and cex do little to improve the looks.

predTree <- predict(modFitTree, newdata = testing, type = "class")
confTree <- confusionMatrix(predTree, as.factor(testing$classe))
confTree

# Create a data frame to store overall accuracies for comparison
Accuracy <- data.frame(matrix(ncol = 3, nrow = 1))
colnames(Accuracy) <- c("DecisionTree", "RandomForest", "GBM")
Accuracy$DecisionTree <- as.numeric(confTree$overall['Accuracy'])


#Random Forest
training_na <- training[complete.cases(training),] #remove a row containing NA.  RandomForest refused to run when there's even a single NA in them.
#This default setting takes more than 50mins to run.  May need to try sampsize=5000, ntree=500 to see the difference.
modFitRF <- train(classe ~., data = training_na, method = "rf")
predRF <- predict(modFitRF, newdata = testing)
confRF <- confusionMatrix(predRF, as.factor(testing$classe))
confRF
Accuracy$RandomForest <- as.numeric(confRF$overall['Accuracy'])

# GBM
modFitGBM  <- train(classe ~ ., data=training_na, method = "gbm",verbose = FALSE)
predGBM <- predict(modFitGBM, newdata = testing)
confGBM <- confusionMatrix(predGBM, as.factor(testing$classe))
confGBM
Accuracy$GBM <- as.numeric(confGBM$overall['Accuracy'])
# Final fit using random forest
predFinal <- predict(modFitRF, newdata=pml_testing)
predFinal

## Try parallel processing
library(doParallel)
cores <- detectCores() # returns 4.  2 cores with 4 logical processors on i7-7500u
registerDoParallel(cores-1) # leave 1 core for CPU ? necessary??
rfControl <- trainControl(allowParallel = TRUE)
modFitRF <- train(classe ~., data = training_na, method = "rf", trControl=rfControl)
#Looks to be working.  CPU hoovering at 42% (compare to 32%) and at times hits 92-96%

