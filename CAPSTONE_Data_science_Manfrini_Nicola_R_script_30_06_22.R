
##CAPSTONE MOVIELENS EXERCISE - MANFRINI NICOLA - R script

# I started by uploading all the data as recomended #

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################


if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


# Below are the instructions given#

## INSTRUCTIONS ##

#you will train a machine learning algorithm using the inputs in one subset to predict movie ratings in the validation set. 
#Your project itself will be assessed by peer grading.

#Predict movie ratings in the validation set (the final hold-out test set) as if they were unknown. 
#RMSE will be used to evaluate how close your predictions are to the true values in the validation set 
#(the final hold-out test set).

# OK, LET'S START #

# I will start sub-partitioning edx into test and training set #
# I will do this to prepare models on the sub-training set and testing them on the sub-test set #

library(dslabs)
test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.2, list = FALSE)

train_set <- edx[-test_index,]
test_set <- edx[test_index,]
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# I'll define RMSE, as I will need it throughout the assessment #

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# let's define mu which is the mean of the overall ratings given in the train set, and will be our starting point for 
#prediction#
mu <- mean(train_set$rating)
# let's define our initial RMSE, technically from here we can only do better #
# model number 1 #
model_1_rmse <- RMSE(test_set$rating, mu)
rmse_results <- (data_frame(method="Just the mean model",RMSE = model_1_rmse ))
# we have a bad 1.0607, we can only improve# 
# starting from the mean of rating values we will develop a linear regression model to predict ratings#
# let's have a look at what we are dealing with to have an idea of the predictors we can use to implement a model on ratings #
names(edx)

## let's begin implementing our model starting from one simple assumption:
# By experience for sure some genres are rated more than others
# First thing let's see if this holds true ##

genres_c <- train_set %>% 
  group_by(genres) %>% 
  summarize(c_g = mean(rating - mu))
genres_c
qplot(c_g, data = genres_c, bins = 10, color = I("black"))

## indeed this seems to be the case ##
#let's prepare a model using this genre bias #
# model number 2

predicted_ratings <- test_set %>% 
  left_join(genres_c, by='genres') %>%
  mutate(pred = mu + c_g) %>%
  .$pred
model_2_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Genres Model",  
                                     RMSE = model_2_rmse ))

# let's check if it is ok #                 
rmse_results
# cool it seems to work#
# we improved a little bit #
# let's see if users also have an effect on ratings, if affermative we will also use them to prepare a model #
user_c <- train_set %>% 
  left_join(genres_c, by='genres') %>%
  group_by(userId) %>%
  summarize(c_u = mean(rating - mu - c_g))
user_c
qplot(c_u, data = user_c, bins = 10, color = I("black"))
# yes also used id inserts a great bias #
## let's implement using users #
# model number 3 # 
predicted_ratings <- test_set %>% 
  left_join(genres_c, by='genres') %>%
  left_join(user_c, by='userId') %>%
  mutate(pred = mu + c_g + c_u) %>%
  .$pred
model_3_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Genres + User Model",  
                                     RMSE = model_3_rmse ))
# Let's see if we actually improved#
rmse_results
#Great, we did we got a 0.942, much better#
# Let's implement using titles, as titles per se could affect movie rating #
#let's see if it is true ##

title_c <- train_set %>% 
  left_join(genres_c, by='genres') %>%
  left_join(user_c, by='userId')%>%
  group_by(title) %>%
  summarize(c_t = mean(rating - mu - c_g - c_u))

qplot(c_t, data = title_c, bins = 10, color = I("black"))


# yes, different titles are rated much differently ##
# let's make a new model then inserting titles
# model number 4 #

predicted_ratings <- test_set %>% 
  left_join(genres_c, by='genres') %>%
  left_join(user_c, by='userId') %>%
  left_join(title_c, by='title') %>%
  mutate(pred = mu + c_g + c_u + c_t) %>%
  .$pred
model_4_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Genres + User + title Model",  
                                     RMSE = model_4_rmse ))
# let's see if we did better #
rmse_results
# this fourth model seems to work really good#

#but let's try a little harder, let's try inserting time, maybe in different periods movies were rated differentially#

# model 5 #
# Let's check time stamps #
train_set$timestamp
# let's convert timestamp to dates using month as unit #
# to do so we need to install the lubridate package #
if(!require(lubridate)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
library(lubridate)

time_c <- train_set %>% 
  left_join(genres_c, by='genres') %>%
  left_join(user_c, by='userId') %>%
  left_join(title_c, by='title') %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "month")) %>%
  group_by(date) %>%
  summarize(c_time = mean(rating - mu - c_g - c_u - c_t))
time_c

# Let's see if time gives any bias #
qplot(c_time, data = time_c, bins = 10, color = I("black"))

# there's a really really small, if any, bias due to time let's test it anyway #
# let's add it to our model #

predicted_ratings <- test_set %>% 
  left_join(genres_c, by='genres') %>%
  left_join(user_c, by='userId') %>%
  left_join(title_c, by='title') %>%
  mutate(date = round_date(as_datetime(timestamp), unit = "month")) %>%
  left_join(time_c, by='date') %>%
  mutate(pred = mu + c_g + c_u + c_t + c_time) %>%
  .$pred
model_5_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Genres + User + title + time month Model",  
                                     RMSE = model_5_rmse ))

rmse_results
# time model using month does not add anything ##
# let's try only time using week on original data to see if there's any bias
time_week_only <- train_set %>% 
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  group_by(date) %>%
  summarize(only_time = mean(rating - mu))
time_week_only

predicted_ratings <- test_set %>% 
  mutate(date = round_date(as_datetime(timestamp), unit = "week")) %>%
  left_join(time_week_only, by= 'date') %>%
  mutate(pred = mu + only_time) %>%
  .$pred
model_6_rmse <- RMSE(predicted_ratings, test_set$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="time_week_only_model",  
                                     RMSE = model_6_rmse ))
rmse_results
# nothing, really bad, so we will stick with model number 4 #
## now let's check on our final dataset  for model number 4 #
#will our model be efficient if produced on edx data and tested on the validation set? ##

mu_edx <- mean(edx$rating) 
genres_c_edx <- edx %>% 
  group_by(genres) %>% 
  summarize(c_g_edx = mean(rating - mu_edx))

user_c_edx <- edx %>% 
  left_join(genres_c_edx, by='genres') %>%
  group_by(userId) %>%
  summarize(c_u_edx = mean(rating - mu_edx - c_g_edx))

title_c_edx <- edx %>% 
  left_join(genres_c_edx, by='genres') %>%
  left_join(user_c_edx, by='userId')%>%
  group_by(title) %>%
  summarize(c_t_edx = mean(rating - mu_edx - c_g_edx - c_u_edx))

predicted_ratings_temp <- validation %>% 
  left_join(genres_c_edx, by='genres') %>%
  left_join(user_c_edx, by='userId') %>%
  left_join(title_c_edx, by='title') %>%
  mutate(pred = mu_edx + c_g_edx + c_u_edx + c_t_edx) %>%
  .$pred
RMSE_final_model <- RMSE(predicted_ratings_temp, validation$rating)
RMSE_final_model

## yes!!! we created a simple model that seems to work#
## cool! ##
# our final model RMSE is more or less 0.871, even better than expected from our train/test sets, it did pretty good!!!
