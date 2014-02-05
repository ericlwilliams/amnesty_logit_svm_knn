## Comparing Logistic Regression, SVM, kNN for predicting follow-up in *all* Amnesty International Urgent Actions (not split by category, yet)
### save_dtm.R
- Run once to generate ./data/amnesty_dtm.RData
	- DTM of Urgent Actions 
	- 99% sparse: ~5165 Urgent Actions, with 3506 inputs from DTM
	- 95% sparse (./data/amnesty_s95_dtm.RData): ~5165 Urgent Actions, with 653 inputs from DTM

### amnesty_model_compare.R
- Compares:
	- Logistic Regression w/ LASSO regularization via 10-fold X-valiation
	- kNN (k = 5-50) 
	- SVM (linear,poly,radial) w/ cost = 1-10