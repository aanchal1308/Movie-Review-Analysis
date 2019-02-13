# Movie-Review-Analysis
This sentiment analysis of movie reviews classifies the movie reviews in classes: positive and negative.
It consists of a simple classifier build using logistic regression algorithm.
Dataset: IMDb movie reviews

Steps for downloading and pre-processing:
1.  Go to [IMDb Reviews](http://ai.stanford.edu/~amaas/data/sentiment/) and click on “Large Movie Review Dataset v1.0”. Once that is complete you’ll have a file called `aclImdb_v1.tar.gz` in your downloads folder.
2. Open a terminal window and `cd` to the directory that you put aclImdb_v1.tar.gz in.
3. `gunzip -c aclImdb_v1.tar.gz | tar xopf -`
4. `cd aclImdb && mkdir movie_data`
5. `for split in train test; do for sentiment in pos neg; do for file in $split/$sentiment/*; do cat $file >> movie_data/full_${split}.txt; echo >> movie_data/full_${split}.txt; done; done; done;`
