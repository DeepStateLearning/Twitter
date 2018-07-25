# Twitter
I'm uisng this space to play around with some attempt to put metrics on sets of twitter accounts, and also to try out some clustering techniques. 

The first attempt is the following:
1)Take a group of users selected according to some criterion, 
2)Collect the set of retweets that these users made
3)Put a distance metric on most commonly retweeted tweets
4)Using the distance metric on tweets, compute Wasserstein metric on users based on their set of retweets
5)Use multidimensional scaling to look at the distance metric in 2d
6)Use Ricci flow to further cluster the data 

As we can see, this can distinguish groups of users quite well: In the first example there's a very clear distinction between pro-Trump and anti-Trump (of course these were selected) 

More about the examples
 
In the first example, we start with a set of political accounts obtained from Steve Kramer:  https://data.world/drstevekramer/twitter-bot-detection-for-walkaway-campaign, we see the above process splits the space quite nicely.  MOre details in the page "Example1.md". 


The following is the 2d picture which are colored by KMeans
![Image of  metric](https://github.com/DeepStateLearning/Twitter/blob/master/RedBlueNoRicci.png)


The following is the 2d picture after some Ricci flow 
![Image of clustered metric](https://github.com/DeepStateLearning/Twitter/blob/master/RBwithRIcci1500.png)

We also looked at the "sources" of the tweets and found no obvious pattern


For the second example, we took the users who had used the hashtag #FreeMariaButina.   The picture "maria.png" is very boring and random.  No obvious information, without comparing these to other groups. This could be because it's actually boring and random or because it's quite focused and we can't see everything else. 

![Image of Maria](https://github.com/DeepStateLearning/Twitter/blob/master/maria.png)



In the third example, we took the accounts that used hashtag #amazonstrike.   These separate into to distinct groups.  You can see by looking closer at the file with the metric "amazon_metric.csv" that the distinction is largely between Spanish la and English.  


![Image of Amazon](https://github.com/DeepStateLearning/Twitter/blob/master/amazonRicci.png)
