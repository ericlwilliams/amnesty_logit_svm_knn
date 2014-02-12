# Equivalent performance with PCA reduction
# Keep loadings st ~90% variance retained
# TODO -  apply to model_compare, only apply PCA to training

library("glmnet") # linear

load("./data/amnesty_s95_dtm.RData")

# aiua.pca <- princomp(aiua.dtm[,2:ncol(aiua.dtm)],cor=TRUE)
aiua.pca <- princomp(aiua.dtm[,2:ncol(aiua.dtm)])# covariance because scales are similar

# summary(p1)
# loadings(p1)

# plot(p1)
# biplot(p1)
# p1$scores
# screeplot(p1) ## identical with plot()
pdf("./figures/pca/aiua_screeplot_10npcs.pdf")
screeplot(aiua.pca, npcs=10, type="lines")
dev.off()

# Retain top  to

aiua.scores <- aiua.pca$scores[,1:10]


# run linear regression
# NOT VALID - only run on training... TODO


# full for xval
full.x <- aiua.scores
full.y <- aiua.dtm[,1]

# split in half for test/training
training.indices <- sort(sample(1:nrow(aiua.dtm), round(0.8*nrow(aiua.dtm))))
test.indices <- which(! 1:nrow(aiua.dtm) %in% training.indices)


train.x <- full.x[training.indices,]
train.y <- full.y[training.indices] # followup

test.x <- full.x[test.indices,]
test.y <- full.y[test.indices] # followup

rm(aiua.dtm)

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

