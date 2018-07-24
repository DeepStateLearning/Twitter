First we started with this file, which is obtained by Steve Kramer 
https://data.world/drstevekramer/social-media-bot-detection-by-paragon-science/workspace/file?filename=all_potential_bots.txt which lists about 5300 accounts that do a lot political activity.  

Step 1: ("store_user_tweets.py")  Go obtain the statuses for all of these users, in an artitrary time window (in this case between tweets 1019098933979580000
and 1015000000000000000 ).   The status data is pickled (I might use it later for more analysis) and the ids of all the retweets by the users is stored in a dataframe: "retweet_ids.csv.
This successfully got user data for about 3000 usernames.  
Step 2. ("metrics.py") Create a similarity metric on the 600 popular retweets. Use this together with Wasserstein metric to get a metric on the users.  Then do Ricci flow on the metric to accentuate the clusters. 

