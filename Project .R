# Load required libraries
library(caret)
library(ROCR)
library(randomForest)
library(nnet)
library(ranger)
library(e1071)
library(pROC)
#######################################################################
##Preprocess features
# Define column classes for reading in the data sets
colClasses <- c(rep("integer", 4), "character", rep("factor", 2),
                "integer", "factor", "integer", "factor", rep("integer", 3),
                rep("factor", 2), "integer", rep("factor", 7), "integer")

# Function of Feature Engineering
Data_Processing <- function(data) {
  #Adding Features
  data$hour <- as.integer(substr(data$time, 1, 2))
  data$timeofday <- as.factor(ifelse(data$hour %in% 6:16, "day",ifelse(data$hour %in% 17:20, "evening", "night")))
  data$weekend <- as.factor(ifelse(data$day %in% 0:4, "No", "Yes"))
  data$plan <- paste0(data$A, data$B, data$C, data$D, data$E, data$F, data$G)
  data$family <- as.factor(ifelse(data$group_size > 2 & data$age_youngest <25 & data$married_couple==1, "Yes", "No"))
  data$agediff <- data$age_oldest-data$age_youngest
  data$individual <- as.factor(ifelse(data$agediff==0 & data$group_size==1,"Yes", "No"))

  # Predict Risk Factors for NA
  dataworisk <- data[is.na(data$risk_factor), ]
  datawrisk <- data[!is.na(data$risk_factor), ]
  fit <- lm(risk_factor ~ age_youngest*group_size+married_couple+homeowner, data=datawrisk)
  pred <- predict(fit, newdata=dataworisk)
  data$risk_factor[is.na(data$risk_factor)] <- round(pred, 0)
  # Car Age more than 25
  data$car_age[data$car_age > 25] <- 25
  # Set NA for duration_previous
  levels(data$C_previous) <- c("1", "2", "3", "4", "none")
  data$C_previous[is.na(data$C_previous)] <- "none"
  data$duration_previous[is.na(data$duration_previous)] <- 0
  return(data)
}

#########################################################################
# read data
train <- read.csv("train.csv", colClasses=colClasses)
train <- Data_Processing(train)

#setting index to split training and testing 
index=1:50000
index2=50001:97009
# trainsub is subset of train with only purchases
trainsub <- train[!duplicated(train$customer_ID, fromLast=TRUE), ]
trainsub1 = trainsub[index,]
testsub=trainsub[index2,]
# trainex is subset of train that without purchases
trainex <- train[duplicated(train$customer_ID, fromLast=TRUE), ]
# trainex2 is last quote before purchase
trainex2 <- trainex[!duplicated(trainex$customer_ID, fromLast=TRUE), ]
trainex2_1=trainex2[index,]
testex2=trainex2[index2,]
# changed is customers changed from last quote
changed <- ifelse(trainsub$plan == trainex2$plan, "No", "Yes")
changedtest=changed[index2]
changedtrain=changed[index]
length(which(changed=="No"))/length(changed)
changelog <- ifelse(trainsub$plan == trainex2$plan, FALSE, TRUE)
trainsub1$changedtrain <- as.factor(changedtrain)
trainex2_1$changedtrain <- as.factor(changedtrain)
testsub$changedtest <- as.factor(changedtest)
testex2$changedtest <- as.factor(changedtest)
changelogtrain=changelog[index]
changelogtest=changelog[index2]
trainsub1$changelogtrain <- changelogtrain
trainex2_1$changelogtrain <- changelog[index]
testsub$changelogtest <- changelog[index2]
testex2$changelogtest<- changelog[index2]

# Compute "Stability" Feature 
customerstability <- trainex %>% group_by(customer_ID) %>%summarise(quotes=n(), uniqueplans=n_distinct(plan),
            stability=(quotes-uniqueplans+1)/quotes)
stability <- customerstability$stability[index]
trainex2_1<-cbind(trainex2_1,stability)
stability <-customerstability$stability[index2]
testex2<-cbind(testex2,stability)
####################################################################################################
##Modeling
# logistic modeling
set.seed(5)
#result=list()
#result2=list()
#folds <- sample(rep(1:5, length = nrow(trainex2_1)))
#for(k in 1:5) {
# fit <- glm(changelogtrain ~ state+cost+A+B+C+D+E+F+G+age_oldest+age_youngest+
#             car_value+car_age+shopping_pt+timeofday+weekend+risk_factor+C_previous+
#            duration_previous+stability, data=trainex2_1[folds!=k, ],
#         family=binomial)
# probs<- predict(fit, newdata=trainex2_1[folds==k, ], type="response",se.fit=F)
#pred <- ifelse(probs>0.52, "Yes", "No")
#result[k]=list(probs)
#result2[k]=list(pred)
#print(mean(pred==trainex2_1$changedtrain[folds==k]))
#}
traindata=cbind(trainex2_1$state,trainex2_1$cost,trainex2_1$A,trainex2_1$B,trainex2_1$C,trainex2_1$D,trainex2_1$E,trainex2_1$F,trainex2_1$G,trainex2_1$age_oldest,trainex2_1$age_youngest,trainex2_1$car_value,trainex2_1$car_age,trainex2_1$shopping_pt,trainex2_1$timeofday,trainex2_1$weekend,trainex2_1$risk_factor,trainex2_1$C_previous,trainex2_1$duration_previous,trainex2_1$stability,changelogtrain)
colnames(traindata)= c("state","cost","A","B","C","D","E","F","G","age_oldest","age_youngest","car_value","car_age","shopping_pt","timeofday","weekend","risk_factor","C_previous","duration_previous","stability","ylabel")
rateofunchanged3=NULL
fit3 <- glm(changelogtrain~ state+cost+A+B+C+D+E+F+G+age_oldest+age_youngest+
              car_value+car_age+shopping_pt+timeofday+weekend+risk_factor+C_previous+
              duration_previous+stability, data=trainex2_1,
            family=binomial)
probs3<- predict(fit3, newdata=trainex2_1, type="response",se.fit=F)
pred3<- ifelse(probs3>0.52, "Yes", "No")
summary(fit3)
result_model3=probs3
result2_model3=pred3
rateofunchanged3=length(which(result2_model3==changedtrain))/50000
rightrate=sum(changedtrain[which(result2_model3==changedtrain)]=="No")/50000
glmpred <- prediction(probs3,traindata[,21])
glmperf <- performance(glmpred, "tpr", "fpr")

###################################################################################################  
# test glm model
probstest<- predict(fit3, newdata=testex2, type="response",se.fit=F)
predtest<- ifelse(probstest>0.5, "Yes", "No")
rateofunchangedtest=length(which(predtest==changedtest))/(length(changelog)-50000)
predictrate=sum(changedtest[which(predtest==changedtest)]=="No")/(length(changelog)-50000)
needtrain=testex2[which(predtest!=changedtest),]

####################################################################################################
#correctly predicted changed "yes" from training set (glm)
needtrain=trainex2_1[which(result2_model3!=changedtrain),]
needtrainpurchase=trainsub1[which(result2_model3!=changedtrain),]
noneedtrain=trainex2_1[which(result2_model3==changedtrain),]
hist(noneedtrain$stability)
hist(needtrain$stability)
mean(noneedtrain$cost)
mean(needtrain$cost)

###################################################################################################
# randomforest
# random forests for predicting "changed"
rf.fit <- ranger(dependent.variable.name="ylabel", data=traindata,write.forest = TRUE)
rf.fit$predictions
rf.pred<- ifelse(rf.fit$predictions>0.52, "Yes", "No")
rf.prob<- sum(rf.pred==changedtrain)/50000
rightrate.rf=sum(changelogtrain[which(rf.pred==changedtrain)]==0)/50000
predroc <- prediction(rf.fit$predictions,traindata[,21])
rfperf <- performance(predroc, "tpr", "fpr")

##########################################################
#Ranfom Forest test
testdata=cbind(testex2$state,testex2$cost,testex2$A,testex2$B,testex2$C,testex2$D,testex2$E,testex2$F,testex2$G,testex2$age_oldest,testex2$age_youngest,testex2$car_value,testex2$car_age,testex2$shopping_pt,testex2$timeofday,testex2$weekend,testex2$risk_factor,testex2$C_previous,testex2$duration_previous,testex2$stability,changelogtest)
colnames(testdata)= c("state","cost","A","B","C","D","E","F","G","age_oldest","age_youngest","car_value","car_age","shopping_pt","timeofday","weekend","risk_factor","C_previous","duration_previous","stability","ylabel")
probstest.rf<- predict(rf.fit, data=testdata, type="response")
predtest.rf<- ifelse(probstest.rf$predictions>0.5, "Yes", "No")
rateofunchangedtest.rf=length(which(predtest.rf==changedtest))/(length(changelog)-50000)
predictrate.rf=sum(changedtest[which(predtest.rf==changedtest)]=="No")/(length(changelog)-50000)

#########################################################
#NNET
nnet.fit<- nnet(x=traindata[,-21], y=traindata[,21], size=10) 
summary(nnet.fit)
nnet.fit$fitted.values
nnet.prob<- sum(nnet.fit$fitted.values==changelogtrain)/50000
rightrate.nnet=sum(changelogtrain[which(nnet.fit$fitted.values==changelogtrain)]==0)/50000
predrocnnet <- prediction(nnet.fit$fitted.values,traindata[,21])
nnetperf <- performance(predrocnnet, "tpr", "fpr")

############################################################
##NNET test
predtestnnet <- predict(nnet.fit,testdata[,-21])
testnnet<- ifelse(predtestnnet>0.5, "Yes", "No")
rateofunchangedtest.nn=length(which(testnnet==changedtest))/(length(changelog)-50000)
predictrate.nn=sum(changedtest[which(testnnet==changedtest)]=="No")/(length(changelog)-50000)

##########################################################
#GBM
library(plyr)
library(dplyr)
fitControl <- trainControl(method = "repeatedcv", number = 4, repeats = 4)
gbmFit1 <- train(as.factor(ylabel) ~ ., data =traindata, method = "gbm", trControl = fitControl,verbose = FALSE)
summary(gbmFit1)
gbmFit1$bestTune
gbmFit1$results
gbm.rate=predict(gbmFit1, traindata ,type= "prob")[,2]
gbm.pred=ifelse(predict(gbmFit1, traindata[,-21] ,type= "prob")[,2]>0.5,"Yes","No")
gbm.prob<- sum(gbm.pred==changedtrain)/50000
rightrate.gmb=sum(changelogtrain[which(gbm.pred==changedtrain)]==0)/50000
gbmpred <- prediction(gbm.rate,traindata[,21])
gbmperf <- performance(gbmpred, "tpr", "fpr")
needtrain.gbm.train<- trainex2_1[which(gbm.pred!=changedtrain),]

############################################################
#GBM Test
test.gbm=predict(gbmFit1, testdata ,type= "prob")[,2]
gbm.pred.test=ifelse(test.gbm>0.5,"Yes","No")
gbm.prob.test<- sum(gbm.pred.test==changedtest)/(length(changelog)-50000)
rightrate.gmb.test=sum(changedtest[which(gbm.pred.test==changedtest)]=="No")/(length(changelog)-50000)
needtest.gbm<- testex2[which(gbm.pred.test!=changedtest),]

#################################################################################################
##Training ROC
plot(glmperf,main="ROC Curves for four methods",col="#feb24c")
plot(rfperf, add = TRUE, col="#31a354")
plot(nnetperf, add = TRUE, col = "#756bb1")
plot(gbmperf, add = TRUE, col = "#de2d26")
legend("topleft",legend=c("Logistic Regression","Random Forest","NNet","GBM"),fill=c("#feb24c","#31a354","#756bb1","#de2d26"), lty=1, bty='n', cex=.75)
#summary(gbmFit1)

#var    rel.inf
#stability                 stability 36.8659787
#shopping_pt             shopping_pt 14.1690179
#cost                           cost 11.0040128
#G                                 G 10.1556102
#C_previous               C_previous  5.6979703
#car_age                     car_age  3.2840662
#F                                 F  2.6749316
#A                                 A  2.5019687
#D                                 D  1.9813796
#C                                 C  1.7885594
#timeofday                 timeofday  1.7001797
#car_value                 car_value  1.6346160
#age_oldest               age_oldest  1.5584479
#age_youngest           age_youngest  1.5290003
#E                                 E  1.4514121
#duration_previous duration_previous  0.9294609
#risk_factor             risk_factor  0.8098901
#B                                 B  0.1339904
#weekend                     weekend  0.1295074
 
##############################################################
#Modeling Step 2
# multinom to predict A (repeat for each option)
a.mn.fit <- multinom(A ~ .-customer_ID-record_type-day-time-location-plan-
                       hour-agediff-state, data=needtrain.gbm.train)
a.mn.pred <- predict(a.mn.fit, newdata=needtrain.gbm.train, type="class")
needtrainsub=trainsub1[which(gbm.pred!=changedtrain),]
confusionMatrix(a.mn.pred, needtrainsub$A)
a.mn.pred = as.matrix(a.mn.pred)


# multinom to predict B (repeat for each option)
b.mn.fit <- multinom(B ~ .-customer_ID-record_type-day-time-location-plan-
                       hour-agediff-state, data=needtrain.gbm.train)
b.mn.pred <- predict(b.mn.fit, newdata=needtrain.gbm.train, type="class")
confusionMatrix(b.mn.pred, needtrainsub$B)
b.mn.pred = as.matrix(b.mn.pred)


# multinom to predict C (repeat for each option)
c.mn.fit <- multinom(C ~ .-customer_ID-record_type-day-time-location-plan-
                       hour-agediff-state, data=needtrain.gbm.train)
c.mn.pred <- predict(c.mn.fit, newdata=needtrain.gbm.train, type="class")
confusionMatrix(c.mn.pred, needtrainsub$C)
c.mn.pred = as.matrix(c.mn.pred)


# multinom to predict D (repeat for each option)
d.mn.fit <- multinom(D ~ .-customer_ID-record_type-day-time-location-plan-
                       hour-agediff-state, data=needtrain.gbm.train)
d.mn.pred <- predict(d.mn.fit, newdata=needtrain.gbm.train, type="class")
confusionMatrix(d.mn.pred, needtrainsub$D)
d.mn.pred = as.matrix(d.mn.pred)


# multinom to predict E (repeat for each option)
e.mn.fit <- multinom(E ~ .-customer_ID-record_type-day-time-location-plan-
                       hour-agediff-state, data=needtrain.gbm.train)
e.mn.pred <- predict(e.mn.fit, newdata=needtrain.gbm.train, type="class")
confusionMatrix(e.mn.pred, needtrainsub$E)
e.mn.pred = as.matrix(e.mn.pred)


# multinom to predict F (repeat for each option)
f.mn.fit <- multinom(F ~ .-customer_ID-record_type-day-time-location-plan-
                       hour-agediff-state, data=needtrain.gbm.train)
f.mn.pred <- predict(f.mn.fit, newdata=needtrain.gbm.train, type="class")
confusionMatrix(f.mn.pred, needtrainsub$F)
f.mn.pred = as.matrix(f.mn.pred)


# multinom to predict G (repeat for each option)
g.mn.fit <- multinom(G ~ .-customer_ID-record_type-day-time-location-plan-
                       hour-agediff-state, data=needtrain.gbm.train)
g.mn.pred <- predict(g.mn.fit, newdata=needtrain.gbm.train, type="class")
confusionMatrix(g.mn.pred, needtrainsub$G)
g.mn.pred = as.matrix(g.mn.pred)


# Combine train
plan_pred = cbind(a.mn.pred,b.mn.pred,c.mn.pred,d.mn.pred,e.mn.pred,f.mn.pred,g.mn.pred)
plan_pred = as.data.frame(plan_pred)
plan_pred$plan <- paste0(plan_pred$V1, plan_pred$V2, plan_pred$V3, plan_pred$V4, plan_pred$V5, plan_pred$V6, plan_pred$V7)

# Compare train
right_count = sum(ifelse(plan_pred$plan==trainsub$plan,1,0))
Accuracy = right_count/length(plan_pred$plan)
Total_count = right_count + rightrate * 50000
Overall_acc = Total_count/50000

#myData = read.csv('train.csv')
#saveRDS(myData,"myData.RDS")


