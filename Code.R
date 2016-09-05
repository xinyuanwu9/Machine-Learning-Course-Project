library(caret)
library(rattle)

# load the file
train <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
testing <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""))

# create the validation set
set.seed(1314)
inTrain <- createDataPartition(y = train$classe, p=0.7, list = FALSE)
training <- train[inTrain, ]
validation <- train[-inTrain, ]
dim(training); dim(validation); dim(testing)

# remove near-zero variables
nzv <- nearZeroVar(training,saveMetrics=TRUE)$nzv
training <- training[, !nzv]
validation <- validation[, !nzv]
testing <- testing[, !nzv]

# remove NA variables
NAv <- sapply(training, function(x) sum(is.na(x)))
unique(NAv)
training <- training[, NAv == 0]
validation <- validation[, NAv == 0]
testing <- testing[, NAv == 0]

# remove the first column for each set 
training <- training[, -1]
validation <- validation[, -1]
testing <- testing[, -c(1, 59)]
dim(training); dim(validation); dim(testing)

# transform the data into the same data type
testing <- rbind(training[, -58], testing)
testing <- tail(testing, n = 20)


# decision trees
set.seed(1314)
Mod1 <- rpart(classe ~ ., data = training)
rpartPred <- predict(Mod1, validation, type = "class")
confusionMatrix(rpartPred, validation$classe)
fancyRpartPlot(Mod1)

# boosting with trees
set.seed(1314)
ctrl <- trainControl(method = "repeatedcv", number = 3, repeats = 1)
Mod2 <- train(classe ~ ., method = "gbm", data = training, 
			trControl = ctrl, verbose = FALSE)
gbmPred <- predict(Mod2, newdata = validation)
confusionMatrix(gbmPred, validation$classe)

# random forest
set.seed(1314)
# much faster than train with "rf" method
Mod3 <- randomForest(classe ~ ., data = training)
rfPred <- predict(Mod3, newdata = validation)
confusionMatrix(rfPred, validation$classe)

Mod4 <- train(classe ~ ., method = "rpart", data = training)
rpartPred2 <- predict(Mod4, newdata = validation)
confusionMatrix(rpartPred2, validation$classe)
