# Save Urgent Action DTM as RData
# Not splitting by category

library(tm)
library(data.table)
library(plyr)

data.path <- "./data/aiua.csv"
sparsity = 0.95

ua.dt <- read.csv(data.path,sep=",",stringsAsFactors=FALSE)

ua.dt <- data.table(ua.dt)
ua.dt<-ua.dt[,list(id=data_id,body=body)]


# wfu is y
id.counts <- data.table(count(ua.dt,"id"))
id.counts[,wfu:=as.numeric(freq!=1)]
ua.dt<-merge(ua.dt,id.counts[2:nrow(id.counts),],by="id")
rm(id.counts)

ua.dt<-ua.dt[!duplicated(ua.dt$id),]

documents <- data.frame(Text = ua.dt$body)

row.names(documents) <- 1:nrow(documents)
corpus <- Corpus(DataframeSource(documents))
corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, c(stopwords('english'),"amnesty","international","information","human","rights","uan@aiusa.org","(m)","amnestyusa.org/urgent/","()","(),"))

aiua.dtm <- DocumentTermMatrix(corpus)
# testing sparcity
# Removes terms which have at least sparse fraction of sparsity 
# (eg. sparse = 0.9 <- each term must appear once in at least 1% of emails)

aiua.dtm <- removeSparseTerms(aiua.dtm,sparse=sparsity)

aiua.dtm <- as.matrix(aiua.dtm)

# first column is y
aiua.dtm <- cbind(ua.dt$wfu,aiua.dtm,aiua.dtm)

# colnames(aiua.dtm) <- c("FollowUp",colnames(ua.dt))
colnames(aiua.dtm)[1] <- "FollowUp"

rm(corpus)
rm(data.path)
rm(documents)
rm(sparsity)
rm(ua.dt)

save.image("./data/amnesty_s95_dtm.RData")







