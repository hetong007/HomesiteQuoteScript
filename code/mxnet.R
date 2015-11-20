library(readr)
library(xgboost)

#my favorite seed^^

cat("reading the train and test data\n")
train <- read_csv("../input/train.csv")
test  <- read_csv("../input/test.csv")

y = train[,3]
x = rbind(train[,-c(1,3)],test[,-1])
trind = 1:nrow(train)
teind = setdiff(1:nrow(x), trind)
id = c(train[,1],test[,1])

save(x,y,trind,teind,id,file='../input/ori.data.rda')

load('../input/ori.data.rda')

# There are some NAs in the integer columns so conversion to zero
#train[is.na(train)]   <- 0
#test[is.na(test)]   <- 0
x$PersonalField84[is.na(x$PersonalField84)] = 1
x$PropertyField29[is.na(x$PropertyField29)] = 0

# cat("train data column names and details\n")
# names(train)
# str(train)
# summary(train)
# cat("test data column names and details\n")
# names(test)
# str(test)
# summary(test)


# seperating out the elements of the date column for the data set
x$month <- as.integer(format(x$Original_Quote_Date, "%m"))
x$year <- as.integer(format(x$Original_Quote_Date, "%y"))
x$day <- weekdays(as.Date(x$Original_Quote_Date))

# removing the date column
x <- x[,-1]

p = ncol(x)
cat.list = NULL
cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in 1:p) {
  if (class(x[[f]])=="character") {
    cat.list = c(cat.list, f)
    x[[f]] <- as.integer(as.factor(x[[f]]))
  }
}

save(x,y,trind,teind,cat.list, file='../input/xgboost.dat.rda')

load('../input/xgboost.dat.rda')

# one hot encoding

val.ind <-sample(trind,2000)
nonval.ind <- setdiff(trind, val.ind)

# Train mxnet

pred1 <- predict(clf, data.matrix(test[,feature.names]))
submission <- data.frame(QuoteNumber=test$QuoteNumber, QuoteConversion_Flag=pred1)
cat("saving the submission file\n")
write_csv(submission, "xgb1.csv")
