from twitter import *
import time
import json
import pickle
import pandas as pd


## This uses the twitter API. I store my info in a file c'onfig.py'

config = {}
execfile("config.py", config)
twitter = Twitter(auth = OAuth(config["access_key"], config["access_secret"], config["consumer_key"], config["consumer_secret"]))



df = pd.read_csv("all_potential_bots.txt")  #create pandas data frame with all screennames. 
df.columns=['user_name']
pobots = list(df.user_name) #list of potential bots


#Next get a bunch of data for each names
MID = 1019098933979580000
SID = 1015000000000000000
def get_timeline(sn):   #This pickles all the trimmed status data, doesn't save a lot of user info
    all_statuses = []
    howmany = 0 
    statuses = twitter.statuses.user_timeline(screen_name=sn, count = 200, max_id = MID, include_rts = True, since_id = SID, trim_user = True, exclude_replies = False)
    all_statuses+=statuses
    if len(all_statuses)==0 : return('empty')
    ids = [k['id'] for k in statuses]
    mid = min(ids)   #this throws error if empty
    time.sleep(.5)
    while(True):
        statuses = twitter.statuses.user_timeline(screen_name=sn, count = 200, max_id = mid, include_rts = True, since_id = SID, trim_user = True, exclude_replies = False)
        all_statuses+=statuses
        ids = [k['id'] for k in statuses]
        if(len(ids))<10: break
        mid = min(ids)
        time.sleep(.4)
        print mid, sn, howmany  
        howmany+=1
    pickle.dump(all_statuses, open('status_data/'+sn+".all_statuses","w"))
    
 
for ss in pobots:
    try: get_timeline(ss)
    except: time.sleep(10)  #theres error when users account is closed or suspended, you get 'not authorized'.  May also want to check you haven't exceeded rate limit.  
    print pobots.index(ss), ss
    time.sleep(.5)


# Now we have the timelines downloaded.   Now we create a dataframe containing indexed by users for which we successfully obtained the timelines.  The data stored will be the ids of all the retweets by that user


def get_pickled_status(sn):  #This returns a list of statuses 
    statuses = pickle.load(open('status_data/'+sn+".all_statuses", "r"))   
    return statuses

df9 = pd.DataFrame(columns = ['screen_name', 'retweeted_ids'])


for a in pobots:
    retweets=[]
    try : stas = get_pickled_status(a)
    except : 
        print "don't have data for ", a
        continue
    for j in stas:
        try: retweets+=[j['retweeted_status']['id']]  #each j is a json object with lots of data.  This simply adds the tweet id if retweeted. 
        except : continue
    df9.loc[len(df9)] = [a, retweets]
    print "success for ", a

pickle.dump(df9, open("retweet_ids.pickle", "w"))  #it's much easier for me to pickle/unpickle dataframes with lists.



