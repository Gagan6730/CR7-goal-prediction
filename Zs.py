#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


df = pd.read_csv('Cristano_Ronaldo_Final_v1/data.csv')


# In[11]:


df.head()


# In[12]:


df.info()


# In[14]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[17]:


df.drop('type_of_shot',axis=1,inplace=True)


# In[18]:


df.info()


# In[19]:


df.drop('type_of_combined_shot',axis=1,inplace=True)


# In[20]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[63]:


df[df['is_goal'].isnull()]


# In[27]:


df.info()


# In[26]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[30]:


df.info()


# In[34]:


df["power_of_shot"].fillna(value=df["power_of_shot"].mean(),inplace=True)


# In[35]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[36]:


df["remaining_min"].fillna(value=df["remaining_min"].mean(),inplace=True)


# In[37]:


df["remaining_sec"].fillna(value=df["remaining_sec"].mean(),inplace=True)


# In[38]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[41]:


df.info()


# In[42]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[49]:


df.drop("team_name",axis=1,inplace=True)

df.drop("home/away",axis=1,inplace=True)
df.drop("date_of_game",axis=1,inplace=True)


# In[50]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[91]:


new_series = pd.Series(df.index.values)
new_series


# In[92]:


df["shot_id_number"]=new_series+1
df.head()


# In[93]:


shot_id_mid=df[df['is_goal'].isnull()]
shot_id_mid


# In[94]:


shot_id=shot_id_mid[["shot_id_number","is_goal"]]
shot_id


# In[98]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[97]:


df.info()


# In[99]:


sns.countplot(x='knockout_match',data=df)


# In[101]:


get_ipython().system('pip install cufflinks')
import cufflinks as cf
cf.go_offline()


# In[103]:


import cufflinks as cf
cf.go_offline()


# In[102]:


df['knockout_match'].iplot(kind='hist',bins=30,color='green')


# In[105]:


ratio=4295/24921
ratio


# In[110]:


import random
def fill_knock_out(cols):
    knock_out=cols[0]
    if pd.isnull(knock_out):
        val=random.random()
        if val<=ratio:
            return 1
        else:
            return 0
    else:
        return knock_out


# In[112]:


df["knockout_match"]=df[["knockout_match"]].apply(fill_knock_out,axis=1)


# In[114]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[121]:


def getgame_season(cols):
    game_season=cols[0]
    if pd.isnull(game_season):
        return 2000
    else:
        list1=game_season.split('-')
        val=int(list1[0])
        return val


# In[123]:


df["game_season"]=df[["game_season"]].apply(getgame_season,axis=1)


# In[124]:


df.head(10)


# In[125]:


df


# In[126]:


df['range_of_shot'].iplot(kind='hist',bins=30,color='green')


# In[127]:


def check_range_of_shot(cols):
    range_of_shot=cols[0]
    if pd.isnull(range_of_shot):
        val=random.randint(1,5)
        return val
    elif range_of_shot=='16-24 ft.':
        return 1
    elif range_of_shot=='8-16 ft.':
        return 2
    elif range_of_shot=='Less Than 8 ft.':
        return 3
    elif range_of_shot=='24+ ft.':
        return 4
    else:
        return 5


# In[128]:


df["range_of_shot"]=df[["range_of_shot"]].apply(check_range_of_shot,axis=1)


# In[129]:


df


# In[130]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[131]:


df['area_of_shot'].iplot(kind='hist',bins=30,color='green')


# In[140]:


new_df=pd.read_csv("Cristano_Ronaldo_Final_v1/data.csv")


# In[141]:


df["area_of_shot"]=new_df["area_of_shot"]
df


# In[142]:


def check_area_of_shot(cols):
    r_ratio=12761/16353
    area_of_shot=cols[0]
    if pd.isnull(area_of_shot):
        r=random.random()
        if r<=r_ratio:
            return 5
        val=random.randint(1,6)
        return val
    else:
        list1=area_of_shot.split('(')
        list2=list1[1].split(')')
        str_val=list2[0]
        if str_val=='R':
            return 1
        elif str_val=='L':
            return 2
        elif str_val=='LC':
            return 3
        elif str_val=='RC':
            return 4
        elif str_val=='C':
            return 5
        else:
            return 6


# In[143]:


df["area_of_shot"]=df[["area_of_shot"]].apply(check_area_of_shot,axis=1)
df


# In[144]:


df['shot_basics'].iplot(kind='hist',bins=30,color='green')


# In[146]:


def check_shot_basic(cols):
    r_ratio=11955/16465
    shot_basic=cols[0]
    if pd.isnull(shot_basic):
        r=random.random()
        if r<=r_ratio:
            return 1
        val=random.randint(1,8)
        return val
    else:
#         list1=shot_basic.split('(')
#         list2=list1[1].split(')')
#         str_val=list2[0]
        if shot_basic=='Mid Range':
            return 1
        elif shot_basic=='Goal Area':
            return 2
        elif shot_basic=='Goal Line':
            return 3
        elif shot_basic=='Penalty Spot':
            return 4
        elif shot_basic=='Right Corner':
            return 5
        elif shot_basic=='Mid Ground Line':
            return 6
        elif shot_basic=='Left Corner':
            return 7


# In[147]:


df["shot_basics"]=df[["shot_basics"]].apply(check_shot_basic,axis=1)
df


# In[149]:


x_test=df[df["is_goal"].isnull()]
x_test


# In[153]:


final_x_test=x_test.drop("is_goal",axis=1)
final_x_test


# In[154]:


x_train=df[df["is_goal"].notnull()]
x_train


# In[155]:


final_x_train=x_train.drop("is_goal",axis=1)
final_x_train


# In[159]:


final_y_train=x_train["is_goal"]
final_y_train


# In[180]:


from sklearn.linear_model import LinearRegression


# In[181]:


lm=LinearRegression()


# In[182]:


lm.fit(final_x_train,final_y_train)


# In[183]:


prediction_lr=lm.predict(final_x_test)


# In[184]:


prediction_lr


# In[185]:


from sklearn.ensemble import RandomForestRegressor
rfc = RandomForestRegressor(n_estimators=100)


# In[186]:


rfc.fit(final_x_train,final_y_train)


# In[170]:


prediction_random_forest=rfc.predict(final_x_test)


# In[171]:


prediction_random_forest


# In[187]:


result_lr=pd.DataFrame(prediction_lr,final_x_test["shot_id_number"],columns=["is_goal"])


# In[188]:


result_lr


# In[189]:


result_lr.to_csv("Linear_regression_result")


# In[190]:


result_random_forest=pd.DataFrame(prediction_random_forest,final_x_test["shot_id_number"],columns=["is_goal"])


# In[191]:


result_random_forest


# In[192]:


result_random_forest.to_csv("Random_forest_result")


# In[ ]:





# In[ ]:




