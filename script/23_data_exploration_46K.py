#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob
import json
import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


file_paths = sorted(glob.glob("../data/user_timeline_46K_parsed/*"), key=lambda x: int(x.split('/')[-1].replace('.json', '')))


# In[ ]:


print(file_paths[:5])


# In[ ]:


print(len(file_paths))


# In[ ]:


d_data_all = pd.DataFrame([])


# In[ ]:


for idx, path in enumerate(file_paths, 1):
    print(idx)
    d_data = pd.read_json(path)
    
    d_agg = d_data.groupby('screen_name').agg({'hashtags': lambda x: sum([len(hashtag) for hashtag in x]),
                                         'user_mentions': lambda x: sum([len(user) for user in x]),
                                         'id_tweet': lambda x: len(x)})
    
    d_agg = d_agg.reset_index()
    
    d_agg['hashtag_per_tweets'] = d_agg.hashtags / d_agg.id_tweet
    d_agg['mention_per_tweets'] = d_agg.user_mentions / d_agg.id_tweet
    
    d_data_all = d_data_all.append(d_agg, ignore_index=True)


# In[ ]:


d_data_all.to_json("../data/user_46K_stats.json", orient='records')


# In[ ]:




