import json
import pandas as pd
import glob
import urllib
import requests
import joblib
import re

from maleo.wizard import Wizard
from tqdm import tqdm
from gensim.matutils import jaccard
from strsimpy.levenshtein import Levenshtein


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
        
class SocialPoliticalModel():    
    def __init__(self, model, spwd):
        self.model = joblib.load(model)
        self.spwd = pd.read_csv(spwd, usecols = ['social political'])
    
        
    def tokenizer(self, text):
        output = []
        list_token = text.split('_')
        list_token = list(filter(None, list_token))
        list_token = [token[1:] if token.startswith('@') else token for token in list_token]

        pattern1 = r"([A-Z][a-z])"
        pattern2 = r"([A-Z]{2,})"
        insert_space = r" \1"

        step1 = [re.sub(pattern1, insert_space, token) for token in list_token]
        step2 = [re.sub(pattern2, insert_space, token).split() for token in step1]

        for i in step2:
            output += i

        return output
    
    def jaccard_sim_feature(self, token, spwd):    
        jaccard_user = []
        for user_token in token:
            jaccard_val = []
            for word in spwd['social political']:
                jaccard_val.append(jaccard(user_token, word))
            jaccard_user.append(min(jaccard_val))
        return min(jaccard_user)
    
    def levenshtein_feature(self, token, spwd):
        levenshtein = Levenshtein()
        levenshtein_user = []
        for user_token in token:
            levenshtein_val = []
            for word in spwd['social political']:
                levenshtein_val.append(levenshtein.distance(user_token, word))
            levenshtein_user.append(min(levenshtein_val))
        return min(levenshtein_user)
        
    def features(self, text):
        token = self.tokenizer(text)
        char_count = len(text)
        jaccard_sim = self.jaccard_sim_feature(token, self.spwd)
        levenshtein_dist = self.levenshtein_feature(token, self.spwd)
        
        return jaccard_sim, levenshtein_dist, char_count
    
    def predict(self, text):
        jaccard, levenshtein, char_count = self.features(text)
        feat_dict = {'jaccard_sim': [jaccard], 
                     'levenshtein_dist': [levenshtein], 
                     'char_count': [char_count]}
        feat = pd.DataFrame.from_dict(feat_dict)
        output = self.model.predict(feat)[0]
        return output


class BuzzerFeatures():
    def __init__(self, data_path, profile_data_path, sp_model):
        self.data_path = data_path
        self.profile_data_path = profile_data_path
        self.sp_model = sp_model
        self.wiz = Wizard()
        self.feat = []
    
    
    def read_json(self, path):
        with open(path, 'r') as file:
            return json.load(file)
    
    def write_json(self, path, data):
        with open(path, 'w') as outfile:
            json.dump(data, outfile)
    
    
    def separate_tweets(self, user_data):
        """Data pada key "tweets" terdiri atas independent & dependent tweet.

        Independent tweet = tweet yang dibuat sendiri (inspirasi sendiri)
        Dependent tweet = tweet yang mengutip/quote tweet org lain (quoted tweet)"""

        list_tweets, list_quoted_tweets = [], []

        for twt in user_data['tweets']:
            list_tweets.append(twt['full_text'])
            try:
                list_quoted_tweets.append(twt['quoted_status']['full_text'])
            except:
                pass
        return list_tweets, list_quoted_tweets
    
    
    def get_all_hashtag(self, list_tweets):   
        all_hashtag = []

        wiz = Wizard()
        twt_hashtag = wiz.get_hashtag(pd.Series(list_tweets))['Hashtag']
        n_twt_use_hashtag = len(twt_hashtag)

        for i in twt_hashtag:
            all_hashtag += i
        return n_twt_use_hashtag, all_hashtag
    
    
    def hashtag_related_feat(self, list_tweets):
        n_twt_use_hashtag, all_hashtag = self.get_all_hashtag(list_tweets)

        if n_twt_use_hashtag != 0:
            ratio = (n_twt_use_hashtag/len(list_tweets))
        else:
            ratio = 0
        return all_hashtag, n_twt_use_hashtag, ratio
    
    
    def get_desc(self, filename, user_data, username_desc):
        username = filename.split('/')[-1][:-5]
        if not username.startswith('@'):
            try:
                desc = username_desc.get(username)[1]
            except:
                desc = ''
        else:
            try:
                desc = user_data['description']
            except:
                desc = ''
        return username, desc
    
    
    def get_media_and_url(self, data):
        media_type = None
        url_link = None

        if 'quoted_status' not in data:
            try:
                media_type = data['extended_entities']['media'][0]['type']
            except:
                pass
            if media_type != 'photo' and data['entities']['urls'] != []:
                url_link = data['entities']['urls'][0]['expanded_url']
        return media_type, url_link
        
        
    def extract_url_title(self, data):
        media_type, url_link = self.get_media_and_url(data)

        if url_link is None:
            content_url = None
        else:
            content_url = url_link
        return media_type, content_url
    
    
    def summary_media_content(self, user_data):
        media_content = [self.extract_url_title(twt) for twt in user_data]
        
        if media_content != []:        
            media_type, content_url = zip(*media_content)
            n_photo = media_type.count('photo')
            n_video = media_type.count('video')
            content_url = [item for item in content_url if item is not None]
        else:
            n_photo, n_video, content_url = None, None, None
        return n_photo, n_video, content_url
    
    
    def feature_extraction(self):
        self.feat = []
        for filename in tqdm(self.data_path):
            user_data = self.read_json(filename)
            
            # Checker
            self.error_code = self.error_code_checker(user_data)
            if self.error_code:
                continue
            
            # Separate tweets
            list_tweets, list_quoted_tweets = self.separate_tweets(user_data)
            # Extract hashtag related features
            if list_tweets:
                all_hashtag, n_twt_use_hashtag, ratio = self.hashtag_related_feat(list_tweets)
            else:
                all_hashtag = []
                n_twt_use_hashtag, ratio = 0, 0
            
            # Get username description
            profile_id = self.read_json(profile_data_path)
            username_desc = {user['screen_name']:(user['name'], user['description']) for user in profile_id}
            username, desc = self.get_desc(filename, user_data, username_desc)
            # Get summary of media content
            n_photo, n_video, content_url = self.summary_media_content(user_data['tweets'])
            
            try:
                name = username_desc.get(username)[0]
            except:
                if user_data['tweets']:
                    name = user_data['tweets'][0]['user']['name']
                else:
                    name = user_data['retweets'][0]['user']['name']
            
            is_name_sp = self.sp_model.predict(name)
            
            # Output
            out = {'username': username,
                   'name': name,
                   'is_name_social_political': int(is_name_sp),
                   'desc': desc,
                   'tweets': list_tweets,
                   'n_tweet': len(list_tweets),
                   'quoted_tweets': list_quoted_tweets,
                   'hashtag': all_hashtag,
                   'n_tweet_use_hashtag': n_twt_use_hashtag,
                   'ratio_tweets_use_hashtag': ratio,
                   'n_photo': n_photo,
                   'n_video': n_video,
                   'content_url': content_url}
            self.feat.append(out)
    
    
    def data_preprocessing(self, data):
        if data:
            data = pd.Series(data)
            out = self.wiz.rm_link(data)
            out = self.wiz.rm_non_ascii(out)
            out = self.wiz.rm_punc(out)
            out = self.wiz.slang_to_formal(out)
            out = self.wiz.rm_stopword(out)
            out = out.astype(str).str.strip()
            out = self.wiz.rm_multiple_space(out)
            out = out.apply(str.lower)
            return out.tolist()
        else:
            return data


    def features(self, processed=False):
        if processed:
            self.feature_extraction()
            for user_feat in tqdm(self.feat):
                try:
                    user_feat['desc'] = self.data_preprocessing(user_feat['desc'])[0]
                    user_feat['tweets'] = self.data_preprocessing(user_feat['tweets'])
                    user_feat['tweets'] = list(filter(None, user_feat['tweets']))
                    user_feat['quoted_tweets'] = self.data_preprocessing(user_feat['quoted_tweets'])
                    user_feat['quoted_tweets'] = list(filter(None, user_feat['quoted_tweets']))
                except:
                    pass
            print('Get clean features')
        else:
            self.feature_extraction()
            print('Get raw features')
    
    
    def error_code_checker(self, user_data):
        error_code = ["401 : account_suspended_or_locked", "404 : account_not_found"]
        if user_data['error_code'] in error_code or user_data['status_count'] == 0:
            keys = ['username', 'name', 'is_name_social_political', 'desc', 
                    'tweets', 'n_tweet', 'quoted_tweets', 'hashtag', 'n_tweet_use_hashtag',
                    'ratio_tweets_use_hashtag', 'n_photo', 'n_video', 'content_url']
            out = {key:None for key in keys}
            self.feat.append(out)
            return True
        else:
            return False
        
if __name__ == "__main__":
    data_path = '../data/data_7200/*.json'
    profile_data_path = '../data/profile_id.json'
    sp_model_path = '../model/social_political_clf.pkl'
    spwd_path = '../data/SPWD.csv'
    
    batch_data = list(chunks(glob.glob(data_path), 1000))
    n_batch = 7
    
    sp_model = SocialPoliticalModel(model=sp_model_path, spwd=spwd_path)
    buzzer = BuzzerFeatures(batch_data[n_batch], profile_data_path, sp_model)
    buzzer.features(processed=True)
    buzzer.write_json(f'../data/dataset/buzzer_features_batch_{n_batch}.json', buzzer.feat)
