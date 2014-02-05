library("ggplot2")
library("glmnet") # linear
library("e1071") # svm
library("class") # knn

do_logit <- FALSE
do_linear_svm <- FALSE
do_radial_svm <- TRUE
do_knn <- FALSE

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
	print("Running logistic analysis with 5-fold x-validation (LASSO)...")
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

if(do_linear_svm) # takes forever
{
	print("Running SVM analysis with linear kernel...")
	linear.svm.fit <- svm(train.x, train.y, kernel = 'linear')
	
	# skip hyperparameter optimization
	predictions <- predict(linear.svm.fit, test.x)
	predictions <- as.numeric(predictions > 0)
	mse <- mean(predictions != test.y)
	print(sprintf("Linear SVM MSE: %.3f",mse))

}

if(do_radial_svm) 
{
	print("Running SVM analysis with radial kernel...")
# ## Compare to Radial SVM
radial.svm.fit <- svm(train.x, train.y, kernel = 'radial')

# skip hyperparameter optimization
predictions <- predict(radial.svm.fit, test.x)
predictions <- as.numeric(predictions > 0)
mse <- mean(predictions != test.y)
print(sprintf("Radial SVM MSE: %.3f",mse))
}


if(do_knn)
{
	print("Running kNN analysis...")
performance <- data.frame()

for (k in seq(1, 8, by = 1))
{
	knn.fit <- knn(train.x,test.x,train.y, k = k)

	predictions <- as.numeric(as.character(knn.fit))
	mse <- mean(predictions != test.y)

	performance <- rbind(performance, data.frame(K = k, MSE = mse))

}

best.k <- with(performance, K[which(MSE == min(MSE))])
best.mse <- with(subset(performance, K == best.k), MSE)
print(sprintf("kNN best (k = %d) MSE: %.3f",best.k,best.mse))

ggplot(performance,aes(x=K,y=MSE)) +
	geom_point() +
	xlab("K") +
	ylab("Miss-classification Error")

ggsave("./figures/aiua_knn_perf.pdf")	

}
