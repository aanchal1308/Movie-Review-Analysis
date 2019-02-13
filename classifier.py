from sklearn.feature_extraction.text import CountVectorizer
from clean_movie_data import reviews_train_clean
from clean_movie_data import reviews_test_clean
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Vectorization
cv = CountVectorizer(binary = True) 		#instance of CountVectorizer class
cv.fit(reviews_train_clean) 				#creates dictionary of the vocab on the corpus
X = cv.transform(reviews_train_clean) 		#to encode each document as a vector
X_test = cv.transform(reviews_test_clean)	

# Build Classifier
target = [1 if i < 12500 else 0 for i in range(25000)]
X_train, X_val, Y_train, Y_val = train_test_split( X, target, train_size = 0.75)

for c in [0.01, 0.05, 0.25, 0.5, 1]:
	lr = LogisticRegression(C = c)
	lr.fit(X_train, Y_train)
	print ("Accuracy for C=%s: %s" % (c, accuracy_score(Y_val, lr.predict(X_val))))

# Train the model
final_model = LogisticRegression(C = 0.05)
final_model.fit(X, target)

print ("Final Accuracy: %s" % accuracy_score(target, final_model.predict(X_test)))
feature_to_coef = {
	word: coef for word, coef in zip(cv.get_feature_names(), final_model.coef_[0])
}

for best_positive in sorted(
	feature_to_coef.items(),
	key = lambda x: x[1],
	reverse = True)[:5]:
	print (best_positive)

for best_negative in sorted(
	feature_to_coef.items(),
	key = lambda x: x[1])[:5]:
	print (best_negative)
