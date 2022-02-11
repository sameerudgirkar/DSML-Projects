#!/usr/bin/env python
# coding: utf-8

# # Netflix Movies and TV Shows

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import seaborn as sns
from numpy import NAN, NaN, nan
plt.figure(figsize=(15,15))
plt.axis('off')
img = plt.imread('data/Netflix_img.jpg')
plt.imshow(img)
plt.show()


# **Business Problem**:To analyze the data and generate insights that could help Netflix decide which type of shows/movies to produce more and to show relevant content to audience and attract more audience and content creators in order to grow the business.

# **Dataset**
# Link: d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/000/940/original/netflix.csv

# - **Show_id**: Unique ID for every Movie / Tv Show
# - **Type**: Identifier - A Movie or TV Show
# - **Title**: Title of the Movie / Tv Show
# - **Director**: Director of the Movie
# - **Cast**: Actors involved in the movie/show
# - **Country**: Country where the movie/show was produced
# - **Date_added**: Date it was added on Netflix
# - **Release_year**: Actual Release year of the movie/show
# - **Rating**: TV Rating of the movie/show
# - **Duration**: Total Duration - in minutes or number of seasons
# - **Listed_in**: Genre
# - **Description**: The summary description

# In[22]:


nf = pd.read_csv('data/netflix.csv')
nf.head(5)


# In[23]:


nf.info()
#Info regarding all thhe columns


# In[24]:


nf.shape


# In[25]:


nf.describe()


# In[26]:


nf.describe(include = 'object').T


# In[27]:


nf.nunique()


# In[28]:


max_rows = 20
pd.set_option("display.max_rows", max_rows)


# In[29]:


nf.size


# In[30]:


nf.columns


# In[31]:


nf.dtypes


# In[32]:


nf[nf.duplicated()]  # No repeated rows.


# In[33]:


nf['type'].value_counts(normalize = True)*100


# In[34]:


nf['type'].value_counts().plot(kind = 'pie',autopct='%.2f')
plt.show()


# In[35]:


# Showing the null values with heatmap


# In[36]:


nf.isnull().sum()/len(nf) *100


# In[37]:


nf.isnull().sum()


# In[38]:


plt.figure(figsize=(10,8))
sns.heatmap(nf.isnull())
plt.show()


# In[39]:


for col in nf.columns:
    
    null_per = nf[col].isna().sum()/len(nf) *100
    if null_per > 0:
        print("{} null percentage: {}".format(col,round(null_per,2)))


# In[ ]:





# **Conclusion** : With the help of basic shape of data, data types of all the attributes and from above heat map, we can conclude that only director, cast and country has significant amount of null values in which director's null values are highest.

# # Ratings specific EDA

# In[40]:


rat=nf['rating'].nunique(dropna=False)
rat


# In[41]:


nf['rating'].unique()


# In[42]:


nf['rating'].value_counts() >6


# In[43]:


nf.shape


# In[44]:


nf['release_year'].value_counts().head(10)


# In[45]:


nf_rating_copy = nf
nf_rat = nf.groupby('rating')[['release_year']].count()
nf_rat = nf_rat.drop(['74 min','84 min','66 min','UR','TV-Y7-FV','NC-17'])
nf_rat.reset_index(inplace = True)
nf_rat.columns = ['rating','release_year counts']
nf_rat


# In[46]:


nf_rating=nf_rating_copy.merge(nf_rat,  how='inner')
nf_rating.head()
nf_rating.shape


# In[47]:


# plt.figure(figsize=(10,7))
sns.countplot(data = nf_rating, x = 'rating', hue = 'type')
plt.xlabel('Modified Ratings')
plt.show()


# In[48]:


pd.crosstab(nf['release_year'] ,nf_rating['rating'] ).plot(kind= 'line',figsize = (10,8),title = 'Movie Ratings Trend')
plt.xlim(2000,2020)
plt.xlabel('Post 2000 Release years')
plt.show()


# **Conclusion**: 
# - I have removed the outliers i.e the values ('74 min','84 min','66 min','UR','TV-Y7-FV','NC-17') those were not contributing enough to the plot. Also, as the values/counts were almost non changing for years less than 2000, I have visualized after 2000.
# - From above line plot, as we can clear see that TV-14(unsuitable for childer under 14) has been decreasing lately due to advancements in internet technology and TV-MA(content for mature adults) has been more preferred now a days owing to lockdown restriction and self-isolations due to COVID. 
# - Hence Netflix should focus more on content related to TV-MA ratings

# In[49]:


# New column for datetime created


# # Date-added specific EDA

# In[50]:


nf['date_time_added'] = pd.to_datetime(nf['date_added'])
nf.head(3)


# 

# In[51]:


nf['date_time_added'].dt.year.value_counts()


# In[52]:


nf['date_time_added'].dt.year.value_counts().plot(kind = 'bar')
plt.show()


# In[53]:


# dt_add_year = pd.Series(nf['date_time_added'] .dropna()).apply(lambda x: int(x.strftime("%Y"))) -- For year


# In[54]:


nf['month']=nf['date_time_added'].dt.month.fillna(-1)


# In[55]:


nf['month']=nf['month'].astype('int64')


# In[56]:


nf.head(3)


# In[57]:


sns.countplot(x='month',hue='type',  data=nf)
plt.show()


# **Conclusion** : 
# - From above bar plot, we can conclude the order of release of the movies/tv shows, where maximumn no. of movies/tv shows are added in year 2019. 
# - Also, we can clear conclude from above count plot that more number of movies are produced as compared to tv shows and in the month of February, less number of movies are being added. Netflix should focus on this aspect as well as why this is hapening and should rule out any possibility of mismanagement

# # Released year specific EDA 

# In[58]:


nf['release_year'].value_counts()


# In[59]:


plt.figure(figsize=(10,8))
sns.histplot(data = nf, x = 'release_year',hue = 'type')
plt.show()


# In[60]:


# plt.figure(figsize=(7,7))
nf_rel_yr_2000 = nf.loc[nf['release_year'] > 2000 , ['release_year','type']]
sns.histplot(data = nf_rel_yr_2000, x = 'release_year', bins = 60)
plt.show()


# In[61]:


nf['release_year'].value_counts().plot(kind = 'line')
plt.show()


# In[62]:


# plt.figure(figsize=(15,10))
nf.loc[nf['release_year'] > 2000 , 'release_year'].value_counts().plot(kind = 'line')
plt.show()


# In[63]:


nf.loc[nf['release_year'] > 2010 , 'release_year'].value_counts().plot(kind = 'line')
plt.show()


# In[64]:


nf_rel_yr_2010 = nf.loc[nf['release_year'] > 2010 , ['release_year','type']]
sns.histplot(data = nf_rel_yr_2010, x = 'release_year', bins = 60)
plt.show()


# In[65]:


sns.kdeplot(nf['release_year'])
plt.xlim(2000,2021)
plt.show()


# **Conclusion** : From above barplot, histplot,kdeplot and lineplot for univariate data, we can conclude that, the movies which have relased dates starts right from 1925(nf.describe().min()) and has an increasing trend till 2018 (which had maximum no. of movies/tv shows released) after which the movies/tv shows saw a drop till 2021.

# # Movies and TV shows specific EDA

# In[66]:


nf.groupby('type').type.count()


# In[67]:


sns.countplot(x = 'type', data =nf)
plt.show()


# In[68]:



sns.displot(data = nf_rel_yr_2000, x = 'release_year', bins = 60, hue = 'type')
plt.xlabel('Post 2000 release years ')
plt.xlim(2000,2021)
plt.show()


# In[69]:


# plt.figure(figsize=(25,25))
sns.displot(data = nf_rel_yr_2000, x = 'release_year', bins = 60, hue = 'type')
plt.xlabel('Post 2010 release years ')
plt.xlim(2010,2021)
plt.show()


# In[70]:


sns.pairplot(data= nf, hue = 'type')
plt.show()


# **Conclusion** : From above pairplot, countplot and distplot we can conclude that there are more number of movies produced as compared to tv shows except in year 2021 where tv shows outnumbered the no. of movies.

# # Country specific EDA

# In[71]:


nf['country'].value_counts(dropna = False)


# In[72]:


type(nf['country'].dropna())


# In[73]:


type(nf['country'][2])


# In[74]:


type(nf['country'][1])


# In[75]:


nf['country'].replace(to_replace=[NaN], value=['NaN'], inplace=True)
nf['country']


# In[76]:


country_2dlist = nf['country'].str.split(',').to_list()
# country_2dlist


# In[77]:


nf_country_copy = nf


# In[78]:


country_list_fin = []
for i in range(len(country_2dlist)):
    country_list = []
    for j in range(len(country_2dlist[i])):
        country_list.append(country_2dlist[i][j].strip().lower())
    country_list_fin.append(country_list)
# country_list_fin


# In[79]:


nf_country_df=pd.DataFrame(country_list_fin,index=nf['title']) # as titles are unique for eachrow, keeping it as index as our primary key
# nf_country_df
nf_country_df1=nf_country_df.stack()
# nf_country_df1
nf_country_df2=pd.DataFrame(nf_country_df1)
nf_country_df2
nf_country_df2.reset_index(inplace = True)
nf_country_df2
nf_country_df2 = nf_country_df2[['title',0]]
nf_country_df2
nf_country_df2.columns = ['title','country_modified']
nf_country_df2
nf_country_copy = nf_country_copy.merge(nf_country_df2, on='title',  how='inner')
nf_country_copy.head(3)


# In[80]:


nf_country_copy.shape


# In[81]:


df = nf['country'].str.split(',',expand=True)
df


# In[82]:


df = nf['country'].str.split(',',expand=True)
df.nunique()


# In[83]:


# country_list = []
country_set_fin = set()
for i in range(12):
    list1 = df.loc[:,i].dropna().to_list()
    list2 = []
    for i in list1:
        list2.append(i.strip())
    country_set = set(list2)
    for i in country_set:
        country_set_fin.add(i)        
country_list = list(country_set_fin)
country_list_fin = country_list[1:]
print(country_list_fin,end = ' ')        


# In[84]:


len(country_list_fin)


# In[85]:


country_list = []

for i in range(12):
    list1 = df.loc[:,i].dropna().to_list()
    list2 = []
    for i in list1:
        list2.append(i.strip())
    for i in list2:
        country_list.append(i)


# In[86]:


country_list_filter = filter(None, country_list) # Handling / Filtering of  None/Empty values from the list which frequency was 7
country_list_filter


# In[87]:


max_rows = 130
pd.set_option("display.max_rows", max_rows)


# In[88]:


country_list_series = pd.Series(country_list_filter).value_counts()


# In[89]:


top_10 = country_list_series.head(10)
top_10


# In[90]:


bottom_38 = country_list_series.tail(38)


# In[91]:


top_10[0]


# In[92]:


top_10_df = pd.DataFrame({'Netflix Reach':top_10})
top_10_df


# In[93]:


top10_dfnewest = top_10_df.reset_index()
top10_dfnewest


# In[94]:


top10_dflatest = top10_dfnewest.rename(columns = {'index':'Top 10 Countries'})
top10_dflatest


# In[95]:


plt.figure(figsize=(12,8))
sns.set_theme(style="darkgrid")
sns.barplot(data = top10_dflatest,x = 'Top 10 Countries', y ='Netflix Reach')
plt.xlabel('Top 10 Country wise Netflix Reach')
plt.show()


# In[96]:


# nf['type'].value_counts().plot(kind = 'pie',autopct='%.2f')
# plt.show()


# **Conclusion 1** : For the EDA on country sepcidic data, as we can see that more than one countries are given in the country column, I have converted them into a single list of unique countries, from which we can conclude than there ate 122 unique countries in the dataset 

# **Conclusion 2** :
# - Previously 831 countries had NaN as the entry field for country. Which is 3rd largest after US and India. I removed them and after converting the messy country data into a series of unique countries with Netflix's movie's or tv shows being watched. 
# - **country_list_series.head(10)** contains the value counts of top 10 countries having reach of Netflix.
# -**country_list_series.tail(38)** contains 38 countries where Netflix has only 1 reach. Netflix should focus on relevant content for these downtrodden countries so that it's reach will increase
# - With the help of barplot, we can see that US is having most no. of Neflix reach in terms of movies/tvshow shown on platform which can be given by **top10_dflatest** dataframe
# - **nf_country_copy** is the updated copy of our original dataframe(nf) w.r.t unique countries

# # Movies/ TV show types (listed_in) specific EDA

# In[97]:


df_show_types = nf['listed_in'].str.split(',',expand=True)
df_show_types


# In[98]:


show_types = []

for i in range(3):
    list1 = df_show_types.loc[:,i].dropna().to_list()
    list2 = []
    for i in list1:
        list2.append(i.strip().lower())
    for i in list2:
        show_types.append(i)

# show_types        


# In[99]:


# show_types_Series_new = pd.Series(show_types)
# show_types_Series_new


# In[100]:


show_types_Series = pd.Series(show_types).value_counts()
show_types_top10 = show_types_Series.head(10)
show_types_top10
# show_types_Series


# In[101]:


show_types_top10.index


# In[102]:


nf_listed_in_copy = nf
df_show_types_2dlist = nf['listed_in'].str.split(',').to_list()
# df_show_types_2dlist


# In[103]:


show_types_list_fin = []
for i in range(len(df_show_types_2dlist)):
    show_types_list = []
    for j in range(len(df_show_types_2dlist[i])):
        show_types_list.append(df_show_types_2dlist[i][j].strip().lower())
    show_types_list_fin.append(show_types_list)
# show_types_list_fin


# In[104]:


nf_show_types_df=pd.DataFrame(show_types_list_fin,index=nf['title']) # as titles are unique for eachrow, keeping it as index as our primary key
# nf_show_types_df
nf_show_types_df1=nf_show_types_df.stack()
# nf_show_types_df1
nf_show_types_df2=pd.DataFrame(nf_show_types_df1)
# nf_show_types_df2
nf_show_types_df2.reset_index(inplace = True)
nf_show_types_df2
nf_show_types_df2 = nf_show_types_df2[['title',0]]
# nf_show_types_df2
nf_show_types_df2.columns = ['title','listed_in_modified']
# nf_show_types_df2
nf_listed_in_copy = nf_listed_in_copy.merge(nf_show_types_df2, on='title',  how='inner')
nf_listed_in_copy.head(3)


# In[105]:


nf_listed_in_copy.shape


# In[106]:


nf_listed_in_copy['listed_in_modified'].nunique()


# In[107]:


show_types_top10[0]


# In[108]:


show_types_top10.index


# In[109]:


show_types_top10_dfnew = pd.DataFrame({'Frequency':show_types_top10})
top10_dfnewest = show_types_top10_dfnew.reset_index()
top10_dflatest = top10_dfnewest.rename(columns = {'index':'Genre'})
top10_dflatest


# In[110]:


# plt.figure(figsize=(12,10))
sns.set_theme(style="darkgrid")
sns.barplot(data = top10_dflatest, y = 'Genre', x='Frequency')
plt.xlabel('Frequecy of top 10 Genres on Netflix')
plt.show()


# In[111]:


sns.set_theme(style="darkgrid")
sns.histplot(top10_dflatest,y = 'Genre', x='Frequency')
plt.xlabel('Distribution count of top 10 Genres on Netflix')
plt.show()


# **Conclusion**:
# - **nf_listed_in_copy** is the updated copy of our original (nf)dataframe w.r.t unique 42 genres of tv shows/movies
# - In the above barplot, we can see the data for top 10 genres(listed_in) and it's frequency. It's eveident that, international movies and dramas are having the major share follwed by comedies and International tv shows. Rest of the genres are having less shows on Netflix with frequency less than 1000 which can be seen with histplot
# - Hence Netflix should focus more on these documentaries -869, action & adventure-859,tv dramas-763,independent movies-756,
# children & family movies 641,romantic movies-616

# # TV Shows specific EDA

# In[112]:


tv = nf.loc[nf.type == 'TV Show']
tv.head(3)


# In[113]:


tv.shape


# **Shows greater than 1 season - 883/2676**

# In[114]:


hit_tv1 = tv.loc[tv['duration'].apply(lambda x: int(x.split(" ")[0]))>1]
hit_tv1.head(2)


# In[115]:


hit_tv1.shape


# In[116]:


hit_tv1.title.head(10)


# **Conclusion** :Out of 2676 TV shows, 883 shows have duration more than 1 Season which means, the first season was a hit and liked by audience hence it's beneficial to go for next season.

# **Shows More than 5 seasons - 100/2676**
# 

# In[117]:


hit_tv2 = tv.loc[tv['duration'].apply(lambda x: int(x.split(" ")[0]))>5]
hit_tv2.head(2)


# **Total 100 similar tv shows can be recommended for more focus w.r.t production**

# In[118]:


hit_tv2.shape


# In[119]:


hit_tv2 = hit_tv2.sort_values(by = 'release_year',ascending=False,ignore_index=True)
hit_tv3 = hit_tv2.loc[:,['title','release_year','duration','country']]
# hit_tv3
# hit_tv2[['title','release_year','duration','country']]


# **Sorting w.r.t year and most seasons**

# In[120]:


# hit_tv3[['Num','Seasons']] = hit_tv3['duration'].apply(lambda x: int(x.split(" ")))
# hit_tv3


# In[121]:


hit_tv3[['Num','Seasons']] = hit_tv3['duration'].str.split(" ",expand = True)
hit_tv3['Num'] = hit_tv3['Num'].astype(int)
# hit_tv3


# In[122]:


hit_tv4 = hit_tv3.sort_values(by = 'Num',ascending=False,ignore_index=True)
hit_tv4.head(5)


# In[123]:


hit_tv4.tail(5)


# In[124]:


top_100_title_array = hit_tv2['title'].to_list()
# top_100_title_array 


# # Movies specific EDA

# In[125]:


mov = nf.loc[nf.type == 'Movie']
mov.head(3)


# In[126]:


mov.shape


# In[127]:


mov_dur = mov.loc[:,['duration','release_year']]
mov_dur = mov_dur.dropna()
mov_dur[['average duration of movie','mins']] = mov_dur['duration'].str.split(" ",expand = True)
mov_dur
# mov_dur = mov_dur[['release_year','average duration of movie']]

# mov_dur['average duration of movie'] = mov_dur['average duration of movie'].astype(int)
# # mov_dur.info()
# # mov_dur['']
# mov_dur.groupby('release_year').mean().plot(kind = 'line')
# plt.show()


# **Conclusions**:
# - Out of 2676 TV shows, 883 shows have duration more than 1 Season which means, the first season was a hit and liked by audience hence it's beneficial to go for next season. Shows More than 5 seasons - 100/2676
# - As the rating includes the age bars and not the ratings of a particular movie or tv show, we can't conclude on what should be the audience most watched and liked shows/movies with such data  hence moving on to more evident column i.e duration
# - As we can see form above line plot that the average duration of movie over the years in last 2 decades has been in the range of 90 - 125 mins, so Netflix should focus on movies with short duration on this range
# - Also, it's eveident that, the average duration has been decresing due to lower attention span of audience owing to increasing use of quick and fast content consumption from the social media, hence Netflix should bring in **short movies** or **short films** for better audience reach
# - For TV shows, I have analysed the data w.r.t duration of the tv shows in seasons. The tv show with maximum duartion will be the show which is liked by audience in general and hence have sorted the TV shows as per shows greater than 1 season and 5 seasons respectively in which the ones > 5 seasons contains top 100 shows in array **top_100_title_array**.

# # Directors specific EDA

# In[128]:


nf['director'].value_counts().head(10)


# In[129]:


nf_director_copy = nf


# In[130]:


director_2d_list = nf['director'].apply(lambda x:str(x).split(',')).to_list()


# In[131]:


# director_2d_list


# In[132]:


director_list_fin = []
for i in range(len(director_2d_list)):
    director_list = []
    for j in range(len(director_2d_list[i])):
        director_list.append(director_2d_list[i][j].strip().lower())
    director_list_fin.append(director_list)
# director_list_fin 


# In[133]:


len(director_list_fin)


# In[134]:


nf_director_df=pd.DataFrame(director_list_fin,index=nf['title']) # as titles are unique for eachrow, keeping it as index as our primary key
nf_director_df.head(2)


# In[135]:


nf_director_df1=nf_director_df.stack()
# nf_director_df1
nf_director_df2=pd.DataFrame(nf_director_df1)
# nf_director_df2
nf_director_df2.reset_index(inplace = True)
nf_director_df2


# In[136]:


nf_director_df2 = nf_director_df2[['title',0]]
# nf_director_df2
nf_director_df2.columns = ['title','director_modified']
nf_director_df2


# In[137]:


nf_director_copy = nf_director_copy.merge(nf_director_df2, on='title',  how='inner')
nf_director_copy.shape


# In[138]:


nf_director_copy.head(3)


# In[139]:


nf_director_top10 = nf_director_copy['director_modified'].value_counts().head(11)
nf_director_top10 = nf_director_top10[1:]
nf_director_top10


# In[140]:


nf_director_top10_dfnew = pd.DataFrame({'Frequency':nf_director_top10})
top10_directornewest = nf_director_top10_dfnew.reset_index()
top10_dirlatest = top10_directornewest.rename(columns = {'index':'Top 10 Directors'})
top10_dirlatest


# In[141]:


sns.set_theme(style="darkgrid")
sns.barplot(data = top10_dirlatest, y = 'Top 10 Directors', x='Frequency')
plt.xlabel('Frequecy of no. of movies/tv shows directed by top 10 Directors on Netflix')
plt.show()


# In[142]:


sns.set_theme(style="darkgrid")
sns.histplot(top10_dirlatest,y = 'Top 10 Directors', x='Frequency')
plt.xlabel('Distribution count of movies/tv shows directed by top 10 Directors on Netflix')
plt.show()


# **Conclusions**:
# - **nf_director_copy** is the updated copy of our original dataframe(nf) w.r.t unique Directors
# - Above barplot and histplot shows the top 10 directors who have their movies on Netflix

# # Cast specific EDA

# In[143]:


nf_cast_copy = nf


# In[144]:


cast_2d_list = nf['cast'].apply(lambda x:str(x).split(',')).to_list()


# In[145]:


cast_list_fin = []
for i in range(len(cast_2d_list)):
    cast_list = []
    for j in range(len(cast_2d_list[i])):
        cast_list.append(cast_2d_list[i][j].strip().lower())
    cast_list_fin.append(cast_list)
# cast_list_fin  


# In[146]:


nf_cast_df=pd.DataFrame(cast_list_fin,index=nf['title']) # as titles are unique for eachrow, keeping it as index as our primary key
nf_cast_df.head(2)


# In[147]:


nf_cast_df1=nf_cast_df.stack()
nf_cast_df1


# In[148]:


nf_cast_df2=pd.DataFrame(nf_cast_df1)
nf_cast_df2


# In[149]:


nf_cast_df2.reset_index(inplace = True)
nf_cast_df2


# In[150]:


nf_cast_df2 = nf_cast_df2[['title',0]]
nf_cast_df2


# In[151]:


nf_cast_df2.columns = ['title','cast_modified']
nf_cast_df2


# In[152]:


nf_cast_copy = nf_cast_copy.merge(nf_cast_df2, on='title',  how='inner')
nf_cast_copy.shape


# In[153]:


nf_cast_copy.head(3)


# In[154]:


nf_cast_copy_new = nf_cast_copy.dropna()
nf_cast_copy_new.head(3)


# In[155]:


nf_cast_top10 = nf_cast_copy['cast_modified'].value_counts().head(11)
nf_cast_top10 = nf_cast_top10[1:]
nf_cast_top10


# In[156]:


# nf_cast_new_top10 = nf_cast_copy_new['cast_modified'].value_counts().head(10)
# nf_cast_new_top10


# In[157]:


nf_cast_top10_dfnew = pd.DataFrame({'Frequency':nf_cast_top10})
top10_castnewest = nf_cast_top10_dfnew.reset_index()
top10_castlatest = top10_castnewest.rename(columns = {'index':'Top 10 Casts'})
top10_castlatest


# In[158]:


sns.set_theme(style="darkgrid")
sns.barplot(data = top10_castlatest, y = 'Top 10 Casts', x='Frequency')
plt.xlabel('Frequecy of no. of movies/tv shows acted in by top 10 Casts on Netflix')
plt.show()


# In[159]:


sns.set_theme(style="darkgrid")
sns.histplot(top10_castlatest,y = 'Top 10 Casts', x='Frequency')
plt.xlabel('Distribution count of movies/tv shows directed by top 10 Directors on Netflix')
plt.show()


# **Conclusions**:
# - **nf_cast_copy** is the updated copy of our original dataframe(nf) w.r.t unique Cast from movies/tv shows
# - Above barplot and histplot shows the top 10 casts who have their movies on Netflix

# # EDA w.r.t more focus on INDIA

# In[160]:


nf_country_copy.head(2)
#country_modified


# In[161]:


nf_India = nf_country_copy[nf_country_copy['country_modified'] == 'india']
nf_India.head(2)


# In[162]:


nf_India.shape


# In[163]:


nf_India_tv = nf_country_copy.loc[(nf_country_copy['type'] == 'TV Show') & (nf_country_copy['country_modified'] == 'india')]
nf_India_tv.head(2)


# In[164]:


nf_India_tv.shape


# In[165]:


nf_India_tv['rating'].value_counts()


# **Conclusion**: 
# - **country_modified** is the cleaned column for all unique countries from the modified and updated dataframe **nf_country_copy** after removing all the outliers and messy data.
# - Total **84 tv shows** from India are there on Netflix

# In[166]:


nf_India_movie = nf_country_copy.loc[(nf_country_copy['type'] == 'Movie') & (nf_country_copy['country_modified'] == 'india')]
nf_India_movie.head(2)


# In[167]:


nf_India_movie.shape


# **Conclusion**: 
# - **country_modified** is the cleaned column for all unique countries from the modified and updated dataframe **nf_country_copy** after removing all the outliers and messy data.
# - Total **962 movies** from India are there on Netflix.

# # Top 10 directors -India

# - **nf_India**: updated dataframe with cleaned data w.r.t countries from (nf_country_copy) dataframe
# - **nf_director_copy** : updated dataframe with cleaned data w.r.t every director

# - **nf_India_movie** : Movies from india dataframe
# - **nf_India_tv**    : TV shows from india dataframe

# In[168]:


nf_director_copy.shape


# In[169]:


nf_director_copy.head(2)


# In[170]:


nf_India_Dir = nf_India


# In[171]:


nf_India_Dir = nf_India_Dir.merge(nf_director_copy, on='title',  how='inner')
nf_India_Dir.shape


# In[172]:


top_10_dir_India = nf_India_Dir['director_modified'].value_counts().head(11)
top_10_dir_India = top_10_dir_India[1:]
top_10_dir_India


# In[173]:


nf_India_Dir['director_modified'].nunique()


# In[174]:


top_10_dir_India_dfnew = pd.DataFrame({'Frequency':top_10_dir_India})
top10_dir_INDnewest = top_10_dir_India_dfnew.reset_index()
top10_dir_INDlatest = top10_dir_INDnewest.rename(columns = {'index':'Top 10 Indian Directors'})
top10_dir_INDlatest


# In[175]:


sns.set_theme(style="darkgrid")
sns.barplot(data = top10_dir_INDlatest, y = 'Top 10 Indian Directors', x='Frequency')
plt.xlabel('Frequecy of no. of movies/tv shows directed by top 10 Indian Director on Netflix')
plt.show()


# In[176]:


nf_India_tv2 = nf_India_tv.sort_values(by = 'release_year',ascending=False,ignore_index=True)
nf_India_tv3 = nf_India_tv2.loc[:,['title','release_year','duration','country_modified','rating']]
nf_India_tv3[['Num']] = nf_India_tv3['duration'].str.split(" ",expand = True)[0]
nf_India_tv3['Num'] = nf_India_tv3['Num'].astype(int)
# nf_India_tv3
nf_India_tv4 = nf_India_tv3.sort_values(by = 'Num',ascending=False,ignore_index=True)
nf_India_tv4.head(10)


# In[177]:


sns.pairplot(data = nf_India_tv4)
plt.show()


# In[178]:


sns.pairplot(data = nf_India_tv4, hue = 'rating')
plt.show()


# In[179]:


nf_India_movie2 = nf_India_movie.sort_values(by = 'release_year',ascending=False,ignore_index=True)
nf_India_movie3 = nf_India_movie2.loc[:,['title','release_year','duration','country_modified','rating']]
nf_India_movie3[['Num']] = nf_India_movie3['duration'].str.split(" ",expand = True)[0]
nf_India_movie3['Num'] = nf_India_movie3['Num'].astype(int)
nf_India_movie4 = nf_India_movie3.sort_values(by = 'Num',ascending=False,ignore_index=True)
nf_India_movie4.head(10)


# In[180]:


nf_India_movie4['rating'].value_counts()


# In[181]:


sns.pairplot(data = nf_India_movie4)
plt.show()


# In[182]:


sns.pairplot(data = nf_India_movie4,hue = 'rating')
plt.show()


# **Conclusion**:
# - 1. Out of total 84 tv shows, the tv shows which are mostly enjoyed and hence has more seasons as compared to others are given in **nf_India_tv** dataframe. As it's eveident from the pairplot for TV shows, Netflix should focus mostly on TV-MA (34) and TV-14 (25) i.e TV shows for mature adults and under 14 years as they are mostly preferred by the audience
# 
# - 2. Out of total 962 Indian movies, most of the movies have duration in 100-150 mins. Netflix should focus on this range while producing more in this range as eveident from above pairplot. Movies with rating TV-14(547) i.e content for children above 14 are mostly preffered as opposed to the overall movie ratings throughout the world where TV-MA(232) is dominant. Netflix should focus more on this aspect with respect to Indian audience.
# 
# - 3. Dataframe **top10_dir_INDlatest** provides top 10 Indian directors. Netflix should reach out to them in order to direct more films.The bar plot is conclusive w.r.t this fact.
