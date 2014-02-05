library("ggplot2")
library("glmnet") # linear
library("e1071") # svm
library("class") # knn

do_logit <- TRUE
do_svm <- TRUE

load("./data/amnesty_s95_dtm.RData")
# aiua.dtm available
# 99% sparse - 5165x3506  
# 95% sparse - 5165x653

set.seed(1)

# full for xval
full.x <- aiua.dtm[,2:ncol(aiua.dtm)]
full.y <- aiua.dtm[,1]

# split in half for test/training
training.indices <- sort(sample(1:nrow(aiua.dtm), round(0.8*nrow(aiua.dtm))))
test.indices <- which(! 1:nrow(aiua.dtm) %in% training.indices)


train.x <- aiua.dtm[training.indices, 2:ncol(aiua.dtm)]
train.y <- aiua.dtm[training.indices,1] # followup

test.x <- aiua.dtm[test.indices, 2:ncol(aiua.dtm)]
test.y <- aiua.dtm[test.indices,1] # followup

rm(aiua.dtm)

# Need stratified sampling?
cat()
print(sprintf("Fraction of *all* UA with follow-up: %.3f", sum(full.y)/length(full.y)))
print(sprintf("Fraction of *training* UA with follow-up: %.3f", sum(train.y)/length(train.y)))
print(sprintf("Fraction of *test* UA with follow-up: %.3f", sum(test.y)/length(test.y)))
cat()


if(do_logit)
{
# Fit logistic regression
# 10-fold cross validation
nfold <- 5 # 5 for testing
cvfit <- cv.glmnet(train.x,train.y,nfolds=nfold,family="binomial",type.measure="class")

pdf("./figures/aiua_logit_lasso_xval_perf.pdf")
plot(cvfit)
dev.off()

# print best miss-classification rate
predictions <- predict(cvfit,test.x, s = cvfit$lambda.min)
predictions <- as.numeric(predictions > 0)
mse <- mean(predictions != test.y)
print(sprintf("Logit best (lambda = %.2e) CV MSE: %.3f",cvfit$lambda.min, mse))
# print(sprintf("Or: %.3f",min(cvfit$)))

}

if(do_svm)
{
	linear.svm.fit <- svm(train.x, train.y, kernel = 'linear')
	
	# skip hyperparameter optimization
	predictions <- predict(linear.svm.fit, test.x)
	predictions <- as.numeric(predictions > 0)
	mse <- mean(predictions != test.y)
	print(sprintf("Linear SVM MSE: %.3f",mse))

}



