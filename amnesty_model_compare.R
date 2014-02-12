library("ggplot2")
library("glmnet") # linear
library("e1071") # svm
library("class") # knn

do_logit <- FALSE
do_svm <- TRUE
svm_kernel <- "linear"
do_knn <- FALSE


#load("./data/amnesty_dtm.RData")
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

if(do_svm) # takes forever
{
	print("Running SVM analysis...")
	# svm.fit <- svm(train.x, train.y, kernel = svm_kernel,type ="C-classification")
	# svm.fit <- svm(train.x, train.y, kernel = svm_kernel)

	svm.df <- data.frame(fu=train.y,train.x)

	# svm.tune <- best.tune(svm,fu ~ .,data=data.frame(fu=train.y,train.x),gamma = 2^(-1:1), cost = 2^(2:4))
	 svm.tune <- tune.svm(fu ~ .,data=svm.df,gamma = 10^(-1:1), cost = 10^(2:4))

	# - sampling method: 10-fold cross validation

	# - best parameters:
	#  gamma cost
	#    0.1 1000

	# - best performance: 0.1488578

	# - Detailed performance results:
	#   gamma  cost     error  dispersion
	# 1   0.1   100 0.1488581 0.006060903
	# 2   1.0   100 0.1685080 0.005473440
	# 3  10.0   100 0.1691895 0.005171579
	# 4   0.1  1000 0.1488578 0.006060880
	# 5   1.0  1000 0.1685080 0.005473441
	# 6  10.0  1000 0.1691899 0.005170841
	# 7   0.1 10000 0.1488581 0.006060960
	# 8   1.0 10000 0.1685080 0.005473444
	# 9  10.0 10000 0.1691900 0.005170874

	 print(summary(svm.tune))

	# svm.fit <- svm(fu~.,data=svm.df,method="C-classification",kernel=svm_kernel,
	# 	cost=svm.tune$best.parameters[[2]],gamma=svm.tune$best.parameters[[1]])
	svm.fit <- svm(fu~.,data=svm.df,method="C-classification",kernel=svm_kernel,
		cost=svm.tune$best.parameters[[2]],gamma=svm.tune$best.parameters[[1]])

    browser()
	# skip hyperparameter optimization
	predictions <- predict(svm.fit, x=test.x)
	predictions <- as.numeric(predictions > 0)
	mse <- mean(predictions != test.y)
	print(sprintf("SVM MSE: %.3f",mse))

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
