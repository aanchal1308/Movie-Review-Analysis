reviews_train = []
for line in open('movie_data/full_train.txt', 'r'):
	reviews_train.append(line.strip())

reviews_test = []
for line in open('movie_data/full_test.txt', 'r'):
	reviews_test.append(line.strip())
