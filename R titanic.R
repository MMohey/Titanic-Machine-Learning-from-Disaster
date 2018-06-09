# Load packages
# 
setwd("~/Downloads/Kaggle Titanic/Titanic")

library('ggplot2')
library('ggthemes')
library('scales')
library('dplyr')
library('mice')
library('randomForest')
library("stringr")
library("e1071")
library("caret")
library("nnet")

# Reading data
train <- read.csv('train.csv', stringsAsFactors = F)
test <- read.csv('test.csv', stringsAsFactors = F)

full <- bind_rows(train, test)

# check data
str(full)

# To extract the title
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)
table(full$Sex, full$Title) # Show a table of title vs Sex
# group rare titles together
rare_title <- c('Capt', 'Col', 'Don', 'Dona', 'Dr', 'Jonkheer', 
                'Lady', 'Major', 'Rev', 'Sir', 'the Countess')
# reassigning similar titles
full$Title[full$Title=='Mlle'] <- 'Miss'
full$Title[full$Title=='Ms'] <- 'Miss'
full$Title[full$Title=='Mme'] <- 'Mrs'
full$Title[full$Title %in% rare_title] = 'Rare Title'
# Show a table of Sex vs Title after rearranging
table(full$Sex, full$Title)

# extracting surname
full$Surname <- sapply(full$Name, function(x) strsplit(x, '[,.]')[[1]][1])
# creating a Family size var
full$Fsize <- full$SibSp + full$Parch + 1
full$Family <-paste(full$Surname, full$Fsize, sep = '_')

# visualise the relationhship between Family size and survival
ggplot(full[1:891,], aes(x=Fsize, fill= factor(Survived))) + 
    geom_bar(stat = 'count', position = 'dodge')+
    scale_x_continuous(breaks = c(1,11))

#grouping family size
full$FsizeD[full$Fsize==1] <- 'singleton'
full$FsizeD[full$Fsize>1 & full$Fsize<5] <- 'small'
full$FsizeD[full$Fsize >4] <- 'large'

#visualising grouped family size vs survival
mosaicplot(table(full$FsizeD, full$Survived), shade = TRUE)

# Create a Deck var 
full$Deck<-factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))

# To check for missing values
sapply(full, function(x) sum(is.na(x) | x ==""))

# To check for the indices of missing value 
which(is.na(full$Embarked) | full$Embarked =="")

# filling a missing value for Embarkment
missing_embark_ind <- full[full$Embarked == '',1]

# checking the missing embrakment rows
full[full$Embarked== '',]

# visulising other passengers with similar profiles
ggplot(full, aes(x= Embarked, y=Fare, fill=factor(Pclass))) + 
    geom_boxplot() + 
    geom_hline(aes(yintercept=80))+
    scale_y_continuous(labels = dollar_format())

# the median for First class with Fare of $80 embarked from C, 
# which is similar to our missing value
full$Embarked[c(62,830)] <- 'C'

# check for missing fare
which(full$Fare =="" | is.na(full$Fare))
full[1044,-4]

# visualising the median fare for 3rd class embarked from S
ggplot(full[full$Pclass=='3' & full$Embarked=='S',], aes(x=Fare)) + 
    geom_density(fill ='red', alpha = 0.3)+
    geom_vline(aes(xintercept = median(Fare, na.rm = TRUE)), color = 'blue',
               linetype = 'dashed', lwd=1)+
    scale_x_continuous(labels = dollar_format())+
    theme_few()

# imputing the missing value
full$Fare[1044] <- 
    median(full[full$Pclass=='3' & full$Embarked=='S',]$Fare, na.rm = TRUE)

# #  trying more feature engineering
# #  
# full$Sex <- if_else(full$Sex=="female", 0,1)
# full$Sex <- as.numeric(as.character(full$Sex))
# 
# full$Embarked[full$Embarked =="S"] <-"0" 
# full$Embarked[full$Embarked =="Q"] <-"1" 
# full$Embarked[full$Embarked =="C"] <-"2" 
# full$Embarked <- as.numeric(full$Embarked)
# 
# full$Deck[full$Deck =="A"] <-"1" 
# full$Deck[full$Deck =="B"] <-"2" 
# full$Deck[full$Deck =="C"] <-"3" 
# full$Deck[full$Deck =="D"] <-"4" 
# full$Deck[full$Deck =="E"] <-"5" 
# full$Deck[full$Deck =="F"] <-"6" 
# full$Deck[full$Deck =="G"] <-"7" 
# full$Deck[full$Deck =="T"] <-"8" 
# full$Deck[full$Deck =="" | is.na(full$Deck)] <-"0" 
# full$Deck <- as.numeric(full$Deck)
# 
# full$FsizeD[full$FsizeD=="small"] <- "0"
# full$FsizeD[full$FsizeD=="singleton"] <- "1"
# full$FsizeD[full$FsizeD=="large"] <- "2"
# full$FsizeD <- as.numeric(full$FsizeD)
# #################
# #################
# use mice to impute missing values for age
sum(is.na(full$Age))
# Change some variables factors into factors
factor_vars <- c("PassengerId", "Pclass", "Sex", "Embarked", "Title", 
                 "Surname", "Family", "FsizeD")
full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))
set.seed(123)
# Build the model, exclude less useful var
mice_mod <- mice(full[, !names(full) %in% c("PassengerId", "Name", "Ticket",
                                            "Cabin", "Family", "Surname",
                                            "Survived", method="rf")])

#save the complete output
mice_output <- complete(mice_mod)

# Plot age distribution of Original data vs MICE output
par(mfrow = c(1,2))
hist(full$Age, freq = F, main = "Age: Original Data", col = "darkgreen",
     ylim = c(0,0.04))
hist(mice_output$Age, freq = F, main = "Age: MICE Output", col = "lightgreen",
     ylim = c(0,0.04))

# replacing the original Age with MICE age
full$Age <- mice_output$Age

# Feature Engineering in the Name length
# full$name_length <- nchar(full$Name)
# 

# create new features to separate Adults from Childeren
# full$male_child[full$Age<33 & full$Sex=="male"] <- "m_child"
# full$male_child[full$Age<33 & full$Sex=="female"] <- "f_child"
# full$male_child[full$Age>=33 & full$Sex=="male"] <- "m_adult"
# full$male_child[full$Age>=33 & full$Sex=="female"] <- "f_adult"
# full$male_child <- factor(full$male_child)


full$Child[full$Age < 18] <- 'Child'
full$Child[full$Age >= 18] <- 'Adult'

full$Mother <- 'Not Mother'
full$Mother[full$Sex == 'female' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss'] <- 'Mother'

full$Child  <- factor(full$Child)
full$Mother <- factor(full$Mother)

# split the data into training and test set

train <- full[1:891,]
test <- full[892:1309,]

# Build the model

set.seed(123)
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch+
                         Fare + Embarked + Title + FsizeD + Child + Mother ,
                         data = train)

svm_model <- train(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch+
                     Fare + Embarked + Title + FsizeD + Child + Mother,
                 data = train,
                 method="svmRadial") 
knn_model <- train(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch+
                       Fare + Embarked + Title + FsizeD + Child + Mother,
                   data = train,
                   method="knn", preProcess = c("center", "scale"), 
                   tuneLength = 10,
                   trControl = trainControl(method = "cv"))

# plot the model error
plot(rf_model, ylim = c(0,0.35))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)

# Get var importance
importance <- importance(rf_model)
var_importance <- data.frame(Variables = row.names(importance),
                             Importance = round(
                                 importance[,"MeanDecreaseGini"],2))
rank_importance <- var_importance %>% 
    mutate(Rank = str_c("#", dense_rank(desc(importance))))

ggplot(rank_importance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
    geom_bar(stat='identity') + 
    geom_text(aes(x = Variables, y = 0.5, label = Rank),
              hjust=0, vjust=0.55, size = 4, colour = 'red') +
    labs(x = 'Variables') +
    coord_flip() + 
    theme_few()

# Prediction
prediction_rf <- predict(rf_model,test)
prediction_svm <- predict(svm_model, test)
prediction_knn <- predict(knn_model, test)


output_rf <- data.frame(PassengerID = test$PassengerId, Survived = prediction)
output_svm <- data.frame(PassengerID = test$PassengerId, 
                         Survived = prediction_svm)
output_knn <- data.frame(PassengerID = test$PassengerId, 
                         Survived = prediction_knn)

write.csv(output_rf, file = 'rf_mod_survived_output.csv', row.names = F)
write.csv(output_svm, file = 'svm_mod_survived_output.csv', row.names = F)
write.csv(output_knn, file = 'knn_mod_survived_output.csv', row.names = F)

