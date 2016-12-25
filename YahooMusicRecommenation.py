#A PySpark Application to Recommend Music Artists to Users
#Dataset: Yahoo! Music User Data in CSV Format
#Development Infrastructure: Databricks Community Edition
#Algorithm: Alternating Least Squares (ALS)


#Load the dataset with ratings into an RDD
ratings_data = sc.textFile("ydata-ymusic-user-artist-ratings-v1_0.csv")
#Load the dataset with artist names into an RDD
names_data = sc.textFile("ydata-ymusic-artist-names-v1_0.csv")

#Parse the ratings data
def parse(line):
    line = line.split("\t")
    line[0]=float(line[0])
    line[1]=float(line[1])
    line[2]=float(line[2])/20
    return (line)
input_mapped= ratings_data.map(parse)

#Split build, train, validate, test data for the modeling
rtrain, rvalidate, rtest = input_mapped.randomSplit([6, 2, 2], seed=0L)
predict_val = rvalidate.map(lambda x: (x[0],x[1]))
predict_test = rtest.map(lambda t: (t[0],t[1]))

#Train the model to obtain best rank
from pyspark.mllib.recommendation import ALS
import math
seed = 5L
iterations = 20
regularization_parameter = 0.1
ranks = [8,12,15,20,30,40,50]
errors = []
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1
best_iteration = -1
for rank in ranks:
    rank_model = ALS.train(rtrain, rank, seed=seed, iterations=iterations,lambda_=regularization_parameter)
    pred_rank = rank_model.predictAll(predict_val).map(lambda r: ((r[0], r[1]), r[2]))
    pred_rates_rank = rvalidate.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(pred_rank)
    error = math.sqrt(pred_rates_rank.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors.append(error)
    err += 1
    print ('For rank %s the RMSE is %s' % (rank, error))
    if error < min_error:
        min_error = error
        best_rank = rank
print ('The best model was trained with rank is %s' % best_rank)

#Train the model to obtain the best iteration
from pyspark.mllib.recommendation import ALS
import math
seed = 5L
iterations = [8,10,15,20,30]
regularization_parameter = 0.1
best_rank = 30
errors = []
err = 0
tolerance = 0.02

min_error = float('inf')

best_iteration = -1
for i in iterations:
    iter_model = ALS.train(rtrain, best_rank, seed=seed, iterations=i, lambda_=regularization_parameter)

    pred_iter = iter_model.predictAll(predict_val).map(lambda r: ((r[0], r[1]), r[2]))
    pred_rates_iter = rvalidate.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(pred_iter)
    error = math.sqrt(pred_rates_iter.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors.append(error)
    err += 1
    print ('For iteration %s the RMSE is %s' % (i, error))
    if error < min_error:
        min_error = error
        best_iterations = i
print ('The best model was trained with iteration is %s' % best_iterations)


#Test the test data based on the obtained best rank and iteration
best_iterations = 10
test_model = ALS.train(rtrain, best_rank, seed=seed, iterations=best_iterations, lambda_=regularization_parameter)
pred_test = test_model.predictAll(predict_test).map(lambda r: ((r[0], r[1]), r[2]))
pred_rates_test = rtest.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(pred_test)
test_error = math.sqrt(pred_rates_test.map(lambda r: (r[1][0] - r[1][1])**2).mean())

print ('For testing data, the RMSE is %s' % (test_error))

#Time taken to build our model on 70/30 split
train, test = input_mapped.randomSplit([7, 3], seed=0L)
from time import time
t0 = time()
complete_model = ALS.train(train, best_rank, seed=seed,iterations=best_iterations, lambda_=regularization_parameter)
elapse_time = time() - t0
print ("Model trained in %s seconds" % round(elapse_time,3))

#Prediction on training the model with the best parameters from previous iterations
pretest = test.map(lambda x: (x[0], x[1]))

pred_split = complete_model.predictAll(pretest).map(lambda r: ((r[0], r[1]), r[2]))
pred_rates_split = test.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(pred_split)
split_error = math.sqrt(pred_rates_split.map(lambda r: (r[1][0] - r[1][1])**2).mean())

print ('For testing data, the RMSE is %s' % (split_error))

#Addition of a new user to test our recommender system
new_user_ID = 0
newratings = [
(0,1000001,4),
(0,1000013,3),
(0,1000059,1),
(0,1000076,5),
(0,1000134,4),
(0,1000214,1),
(0,1000203,5),
(0,1000256,3)]
ratings = sc.parallelize(newratings)

ratings_new= input_mapped.union(ratings)
from time import time
t0 = time()
new_ratings_model = ALS.train(ratings_new, best_rank, seed=seed,iterations=best_iterations, lambda_=regularization_parameter)
time_taken = time() - t0
print ("New model trained in %s seconds" % round(time_taken,3))

#parse the dataset with names
def parse_name(line):
    line=line.split("\t")
    line[0]=float(line[0])
    return line
artist_name=names_data.map(parse_name)

#Get the artists, the new user has not yet rated
new_user_ratings_ids = ratings.map(lambda x: x[1]).collect()
new_user_unrated_music = artist_name.map(lambda x: (new_user_ID, x[0])).filter(lambda artist: artist[1] not in new_user_ratings_ids)

#Build an rdd with the new user's unrated music and predict the scores the user might have rated
new_user_recommendations = new_ratings_model.predictAll(new_user_unrated_music)

new_user_recommendations_rating = new_user_recommendations.map(lambda x: (x.product, x.rating))
user_recommendations_complete = new_user_recommendations_rating.join(artist_name).map(lambda x: (x[0],x[1]))

user_recommendations = user_recommendations_complete.map(lambda r: r[1]).sortByKey(ascending=False)

#Artist recommendation for the new user
final_recommend=user_recommendations.map(lambda x: (x[1],x[0]))
print ("Top 5 Artists Recommended for User Id:%s and their Predicted Ratings." % new_user_ID)
final_recommend.take(5)
