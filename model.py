#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


movie_data=pd.read_csv('movies.csv')


# In[3]:


movie_data.head(10)


# In[4]:


import re


# In[12]:


movie_data['genres']=movie_data['genres'].str.replace('|',',')


# In[13]:


movie_data.head()


# In[14]:


movie_data.info()


# In[18]:


movie_data['title']=movie_data['title'].str.replace('(',' ')
movie_data['title']=movie_data['title'].str.replace(')',' ')


# In[19]:


movie_data.head()


# In[28]:


re.sub('[^a-bA-Z]',' ',movie_data['title'][0])


# In[35]:


movie_data['year']=movie_data['title']


# In[65]:


for i in range(len(movie_data)):
    movie_data['year'][i]=re.findall(r'[0-9]+',movie_data['title'][i])


# In[66]:


movie_data.head()


# In[80]:


type(movie_data['year'][0][0])


# In[85]:


for i in range(len(movie_data)):
    print(movie_data['year'][1][0])
    


# In[86]:


movie_data['year of movie']=movie_data['movieId']
for i in range(len(movie_data)):
    movie_data['year of movie'][i]=movie_data['year'][i][0]


# In[87]:


movie_data.head()


# In[88]:


movie_data.drop(columns=['year'],inplace=True)


# In[89]:


movie_data['year']=movie_data['year of movie']


# In[90]:


movie_data.drop(columns=['year of movie'],inplace=True)


# In[91]:


movie_data.head()


# In[94]:


for i in range(len(movie_data)):
    movie_data['title'][i]=re.sub('[0-9]','',movie_data['title'][i])
    


# In[95]:


movie_data


# In[96]:


movie_data.tail(10)


# In[97]:


data=pd.read_csv('movies.csv')


# In[98]:


data.tail(10)


# In[99]:


data.isnull().sum()


# In[101]:


ratings=pd.read_csv('ratings.csv')


# In[102]:


ratings.head()


# In[103]:


dataset=ratings.pivot(index='movieId',columns='userId',values='rating')
dataset


# In[104]:


dataset.fillna(0,inplace=True)


# In[105]:


dataset.head()


# In[127]:


no_user_voted=ratings.groupby('movieId')['rating'].agg('count')


# In[111]:


no_user_voted


# In[126]:


no_movies_voted = ratings.groupby('userId')['rating'].agg('count')


# In[117]:


h=pd.DataFrame(no_movies_voted)


# In[118]:


h.columns


# In[119]:


no_movies_voted


# In[114]:


ratings.describe()


# In[115]:


ratings.shape


# In[121]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[138]:


sns.displot(no_movies_voted)


# In[ ]:


sns.s


# In[128]:


no_movies_voted.shape


# In[129]:


no_user_voted.shape


# In[144]:


f,ax = plt.subplots(1,1,figsize=(16,4))

plt.scatter(no_user_voted.index,no_user_voted,color='yellow')
plt.axhline(y=10,color='r')
plt.xlabel('MovieId')
plt.ylabel('No. of users voted')
plt.show()


# In[145]:


dataset.head()


# In[ ]:





# In[147]:


no_user_voted[no_user_voted>10]


# In[150]:


f,ax = plt.subplots(1,1,figsize=(16,4))

plt.scatter(no_movies_voted.index,no_movies_voted,color='yellow')
plt.axhline(y=50,color='r')
plt.xlabel('userId')
plt.ylabel('No. of votes by each user')
plt.show()


# In[151]:


dataset.head()


# In[152]:


dataset=dataset.loc[no_user_voted[no_user_voted>10].index,:]


# In[154]:


dataset


# In[156]:


dataset=dataset.loc[:,no_movies_voted[no_movies_voted>50].index]


# In[157]:


dataset


# In[158]:


from scipy.sparse import csr_matrix
csr_matrix_data=csr_matrix(dataset.values)


# In[163]:


csr_matrix_data


# In[166]:


csr_matrix_data.todense()
print(csr_matrix_data)


# In[160]:


dataset.reset_index(inplace=True)


# In[167]:


from sklearn.neighbors import NearestNeighbors


# In[168]:


KNN=NearestNeighbors(metric='cosine',algorithm='brute',n_neighbors=20,n_jobs=-1)
KNN.fit(csr_matrix_data)


# In[223]:


def get_movie_recommendation(movie_name,model):
    n_movies_to_reccomend = 10
    movie_list = data[data['title'].str.contains(movie_name)]  
    if len(movie_list):        
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx =dataset[dataset['movieId'] == movie_idx].index[0]
        distances , indices = model.kneighbors(csr_matrix_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = dataset.iloc[val[0]]['movieId']
            idx = data[data['movieId'] == movie_idx].index[0]
            recommend_frame.append({'Title':data.iloc[idx]['title'],'Distance':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
        return df
    else:
        return "No movies found. Please check your input"


# In[224]:


get_movie_recommendation('Iron Man',KNN)


# In[225]:


import pickle
with open('KNN_pickle.pkl','wb') as file:
    pickle.dump(KNN,file)


# In[226]:


with open('get_recommended_movies.pkl','wb') as file:
    pickle.dump(get_movie_recommendation,file)


# In[ ]:





# In[227]:


import dill
with open('KNN_pickle_dill','wb') as file:
    pickle.dump(KNN,file)
    
with open('get_recommended_movies_dill','wb') as file:
    pickle.dump(get_movie_recommendation,file)    


# In[ ]:




