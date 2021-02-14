import os
import glob
import json
import time
import pickle

import tweepy
import pandas as pd
import concurrent.futures
from tqdm import tqdm, notebook

from itertools import compress 
from datetime import datetime

data_dir_path = '../data'
key_dir_path = '../keys'

key_paths = glob.glob(os.path.join(key_dir_path, '*'))
key_paths = [key.replace('\\', '/') for key in key_paths]

class Friends():
    def __init__(self, keys_paths):
        self.keys = self.read_key(keys_paths)
        self.apis = self.auth_twitter()
        self.api_statuses = [True] * len(self.apis)
        self.setup()
        
    def setup(self):
        paths = ["../data/profile", "../data/following", "../data/supports", "../data/user_timeline_46K"]
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
        
    def read_key(self, key_paths):
        return [pickle.load(open(path, 'rb')) for path in key_paths]
    
    def read_csv(self, path):
        d_data = pd.read_csv(path)
        return d_data
    
    def read_json(self, path):
        with open(path, 'r', encoding="utf-8") as file:
            data_dict = json.load(file)
        
        return data_dict
    
    def auth_twitter(self):
        api_list = []
        for key in self.keys:
            auth = tweepy.OAuthHandler(key["api_key"], key["api_secret_key"])
            auth.set_access_token(key["access_token"], key["access_token_secret"])
            api = tweepy.API(auth)
            
            api_list.append(api)
            
        return api_list
    
    def get_free_token(self):
        idx_tokens = list(compress(range(len(self.api_statuses)), self.api_statuses))
        
        if len(idx_tokens) > 0:
            index = idx_tokens[0]
            return self.apis[index], index
        else:
            return None, None

    def output(self, data, path_dir, filename):
        
        try:
            with open(os.path.join(path_dir, filename + '.json'), 'w') as f:
                f.write(json.dumps(data))
        except:
            try:
                with open(os.path.join(path_dir, filename + '.json'), 'w') as f:
                    f.write(data)
            except:
                pickle.dump(data, open(os.path.join(path_dir, filename + '.pkl'), 'wb'))

                
    def limit_handled(self, cursor):
        while True:
            try:
                yield cursor.next()
            except tweepy.RateLimitError:
                print('\tRateLimit', datetime.today().strftime("\t%H:%M:%S %d-%m-%Y"))
                time.sleep(15 * 60)
            except tweepy.TweepError as e:
                msg = e
                if "Failed to send request" in msg.reason:
                    pass
                elif '429' in msg.reason:
                    print('\tRateLimit', datetime.today().strftime("\t%H:%M:%S %d-%m-%Y"))
                    time.sleep(15 * 60)
                else:
                    return
            except StopIteration:
                return
    
    def get_following(self, username, api, index_token):
        user_follower_dict = {username: []}
        
        for follower in self.limit_handled(tweepy.Cursor(api.friends, id=username).items()):
            user_follower_dict[username].append(follower.screen_name)
            self.output(follower._json, '../data/profile', follower.screen_name)
            
        self.output(user_follower_dict, '../data/following', username)
        
        self.api_statuses[index_token] = True
        
    def get_user_timeline(self, username, api, index_token):
        all_tweets = []
        for tweet in self.limit_handled(tweepy.Cursor(api.user_timeline, username).items(300)):
            all_tweets.append(tweet._json)
        
        self.output(all_tweets, '../data/user_timeline_46K', username)
            
        self.api_statuses[index_token] = True
        
friends = Friends(key_paths)

def calculate_time(start, end):
    duration = end - start
    m = int(duration / 60)
    s = int(duration % 60)
    
    return m, s

def get_user_first_tweet(usernames):
    
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for idx, username in enumerate(usernames, 1):
            while True:
                api, index_token = friends.get_free_token()
                if index_token is not None:
                    friends.api_statuses[index_token] = False
                    executor.submit(friends.get_user_timeline, username, api, index_token)
                    break
                    
            if idx == 10:
                print(idx)
    
    end = time.perf_counter()
    m, s = calculate_time(start,end)
    print("Time:", m, s)
    
if __name__ == '__main__':
    with open('../data/supports/46K_users.json', 'r') as f:
        usernames = json.load(f)
    
    get_user_first_tweet(usernames[:20])