#!/usr/bin/env python
# coding: utf-8

# # AeroFit Treadmill Business Case Study

# **Business Problem** - The market research team at AeroFit wants to identify the characteristics of the target audience for each type of treadmill offered by the company, to provide a better recommendation of the treadmills to the new customers. The team decides to investigate whether there are differences across the product with respect to customer characteristics.
# 
# - To perform descriptive analytics to create a customer profile for each AeroFit treadmill product by developing appropriate tables and charts.
# - To construct two-way contingency tables for each AeroFit treadmill product and to compute all conditional and marginal probabilities along with their insights/impact on the business.

# **About Aerofit** - Aerofit is a leading brand in the field of fitness equipment. Aerofit provides a product range including machines such as treadmills, exercise bikes, gym equipment, and fitness accessories to cater to the needs of all categories of people.

# **Dataset** - The company collected the data on individuals who purchased a treadmill from the AeroFit stores during the prior three months.The dataset has the following features:
# 
# - Product Purchased:	KP281, KP481, or KP781
# - Age              :  In years
# - Gender           :	Male/Female
# - Education        :	In years
# - MaritalStatus    :	Single or partnered
# - Usage            :	The average number of times the customer plans to use the treadmill each week.
# - Income           :	Annual income (in USD)
# - Fitness          :	Self-rated fitness on a 1-to-5 scale, where 1 is the poor shape and 5 is the excellent shape.
# - Miles            :	The average number of miles the customer expects to walk/run each week
# 
# **Product Portfolio** -
# - The KP281 is an entry-level treadmill that sells for USD 1,500.
# - The KP481 is for mid-level runners that sell for USD 1,750.
# - The KP781 treadmill is having advanced features that sell for USD 2,500.

# **Importing required packages**

# In[742]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
import warnings
warnings.filterwarnings("ignore")


# **Loading data into Dataframe**

# In[743]:


# Creating a deep copy and a shallow copy inorder to work on outliers and other messy data if any.


# In[744]:


df = pd.read_csv('data/aerofit_treadmill.csv')
df_dcopy = df.copy(deep=True)
df_scopy = df.copy(deep=False)
df.head()


# In[745]:


df.shape


# In[746]:


df.info()


# In[747]:


df.describe()


# In[748]:


df.isna().sum()/len(df) *100


# In[749]:


df.duplicated().sum()


# In[750]:


characteristics = df.columns.values
for i in characteristics :
    print(i,': ',df[i].unique())
    print()


# In[751]:


# Changing datatype of Gender, MaritalStatus and Product from Object to Category.
characteristics_catg = ['Gender', 'MaritalStatus', 'Product']
for i in characteristics_catg:
    df[i] = df[i].astype("category")
df.info()


# **Observations** :
# - We can conclude from above that, No null & duplicate value found in features.
# - There are 3 different products in this dataset (KP281', 'KP481' ,'KP781').
# - Age of customers range from 18 to 50.
# - Education ranges from 12 to 21 (years).
# - There are both Singles and Partenered as buyer.
# - Usage ranges from 2 to 7 (days/week).
# - Fitness level of customers ranges from 1-5.
# - By changing the dtype from object to category, we are reducing the memory usage.

# # **Outliers detection and removal**

# In[752]:


#Boxplot for Products and the Income of customers purchasing those products
sns.boxplot(data=df, x = 'Product', y = 'Income')
plt.show()


# **Observations** : 
# - KP781 Treadmill with advanced features is preffered by the customers with higher income.
# - KP281 Treadmill with the lowest cost and basic features is preffered by the customers with lower income and the KP481 product with moderate features are liked by the customers with upper bracket of low - moderate income group.
# 
# **Inference** : 
# - There aren't any significant outliers for Products and the Income of customers purchasing those products. So no need for outlier removal here.
# - The target audience for KP781 Treadmill should be the higher income group. So the sales team must focus on this range.

# **1.Outlier Handling for Income**:

# In[753]:


#Boxplot for Income of customers purchasing products before outlier removal
sns.boxplot(data=df, x = 'Income')
plt.show()


# In[754]:


df['Income'].mean()


# In[755]:


#I have used shallow copy of our dataframe for storing it's modified version after removing autliers
q1=df['Income'].quantile(.25)
q2=df['Income'].median()
q3=df['Income'].quantile(.75)
iqr=q3-q1 
df_scopy=df[(df['Income']>q1-1.5*iqr)&(df['Income']<q3+1.5*iqr)]
df_scopy.shape


# In[614]:


df.shape


# In[756]:


#Boxplot for Income of customers purchasing products after outlier removal
sns.boxplot(data=df_scopy, x = 'Income')
plt.show()


# In[761]:


#Boxplot for Gender and the Income of customers purchasing products before outlier removal
sns.boxplot(data=df_scopy, x = 'Gender', y = 'Income')
plt.show()


# In[762]:


df.groupby('Gender')['Income'].mean() # Mean before outlier removal


# In[763]:


#Boxplot for Gender and the Income of customers purchasing products after outlier removal
sns.boxplot(data=df_scopy, x = 'Gender', y = 'Income')
plt.show()


# In[764]:


df_scopy.groupby('Gender')['Income'].mean() # Mean before outlier removal


# **Observations**:
# - After outlier removal for income, 19 rows are deleted and in order to draw some insights from the original data in future, stored the modied data in it's shallow copy - **df_scopy**
# - In the boxplot, we can clearly see that most of the outliers are removed and the data is now ready for further analysis and inferences.
# 

# **2.Outlier Handling for Miles:**

# In[765]:


df_scopy1 = df
sns.boxplot(data = df_scopy1, x = 'Miles')
plt.show()


# In[766]:


#I have used shallow copy of our dataframe for storing it's modified version after removing autliers
q1=df_scopy1['Miles'].quantile(.25)
q2=df_scopy1['Miles'].median()
q3=df_scopy1['Miles'].quantile(.75)
iqr=q3-q1 
df_scopy1=df_scopy1[(df_scopy1['Miles']>q1-1.5*iqr)&(df_scopy1['Miles']<q3+1.5*iqr)]
df_scopy1.shape


# In[767]:


sns.boxplot(data = df_scopy, x = 'Miles')
plt.show()


# **Observations**:
# - After outlier removal for **Miles**, 13 rows are deleted and in order to draw some insights from the original data in future, stored the modied data in it's shallow copy - **df_scopy1**
# - In the boxplot, we can clearly see that most of the outliers are removed and the data is now ready for further analysis and inferences.
# - As of now we will be restricting drawing any insights from df_scopy1 and will be foxusing on df_scopy i.e DF obtained after handing outliers on Income column

# # EDA - Univariate Analysis

# **1.Numerical features**

# In[768]:


df_scopy.head()


# In[769]:


new_mean= round(df_scopy.mean(),2)
new_mean


# In[770]:


new_median = df_scopy.median()
new_median


# In[771]:


# Difference in the mean and median of Income before removing outliers
diff_org = round(df['Income'].mean()-df['Income'].median(),2)
diff_org


# In[772]:


# Difference in the mean and median of Income after removing outliers
diff_new = round(df_scopy['Income'].mean()-df_scopy['Income'].median(),2)
diff_new


# In[773]:


diff_in_income = round((diff_new/diff_org) *100,2)
diff_in_income


# **Inference** : From above, we can infer that, there's a 7.31% correction in the Income data after removing outliers as we can see that the difference in the mean and median has decreased from 3123 to 228. Hence the new dataframe i.e **df_scopy**is more suitable for carrying further analysis w.r.t income and gender related cases.

# In[774]:


#EDA on Univariate Numerical variables
def num_feat(col_data):
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
    sns.histplot(col_data, kde=True, ax=ax[0])
    ax[0].axvline(col_data.mean(), color='y', linestyle='--',linewidth=2)
    ax[0].axvline(col_data.median(), color='r', linestyle='dashed', linewidth=2)
    ax[0].axvline(col_data.mode()[0],color='g',linestyle='solid',linewidth=2)
    ax[0].legend({'Mean':col_data.mean(),'Median':col_data.median(),'Mode':col_data.mode()})
    
    sns.boxplot(x=col_data, showmeans=True, ax=ax[1])
    plt.tight_layout()


# In[775]:


num_cols = df.select_dtypes('int64').columns.values
num_cols


# In[776]:


for i in num_cols:
    num_feat(df[i])


# **Observations**:
# 
# **1.Age**
# - Age is skewed towards right.
# - Customers buying treadmill after age of 40 and before 20 are very less.
# - There are few outliers (higher end).
# 
# **2.Education**
# - Most customers have 16 years of Education.
# - There are few outliers (higher end).
# 
# **3.Usage**
# - Majority of users prefers to use Treadmills 3-4 times/week.
# - There are few outliers (higher end).
# 
# **4.Fitness**
# - Most customers have 3-3.5 fitness rating (moderate fit).
# - Very few customers that uses treadmill have low score i.e 1.
# 
# **5.Income**
# - Income is skewed toward right.
# - Most customers have income less than 70k.
# - **Significant no. of Outliers (higher end) are present** as there are very few persons who earn >80k. This makes us mandatory to handle outliers which has been taken care in the first case. Shallow copy of our dataframe(**ds_scopy**) consists of modified data after dealing with ouliers.
# 
# **6.Miles**
# - Miles is skewed towards right.
# - Customers run on an average 80 miles per week.
# - **Significant no. of Outliers (higher end) are present**, where customers are expecting to run more than 200 miles per week.This makes us mandatory to handle outliers which has been taken care in the first case. Shallow copy of our dataframe(**ds_scopy1**) consists of modified data after dealing with ouliers.

# **2.Catagorical features:**

# In[777]:


df.head()


# In[778]:


Product_Price = {'KP281' : '1500',
                'KP481' : '1750',
                'KP781' : '2500'}


# In[779]:


df['Unit Product Price'] = df['Product'].replace(to_replace = Product_Price )
df['Unit Product Price'].value_counts()


# In[780]:


price = df['Unit Product Price'].unique()
price


# In[781]:


quantity = df['Unit Product Price'].value_counts()
quantity


# In[782]:


for i in range(len(price)):
    tot_sale_USD = quantity[i] * int(price[i])
    print("Total sales for Aerofit treadmills of unit price ${} is ${}".format(int(price[i]),tot_sale_USD))    


# In[784]:


df["Income"].min(),df["Income"].max()


# In[785]:


bins=[0,14,24,40,64,100]  
bins_income = [29000, 40000, 60000, 80000,105000]
label1=['0-14','15-24','25-40','41-64','65-100']
label2=['Children',"Youth", "Young Adults","Old Adults","Seniors"]
label3 = ['Low Income','Moderate Income','High Income','Very High Income']
df['Age Groups']=pd.cut(df['Age'],bins,labels = label1)
df['Age Category']=pd.cut(df['Age'],bins,labels = label2)
df['Income Groups'] = pd.cut(df['Income'],bins_income,labels = label3)
df.head()


# In[786]:


df['Age Category'].value_counts()


# In[787]:


# Change on shallow copy(df_scopy) as well for future analysis.
df_scopy['Unit Product Price'] = df_scopy['Product'].replace(to_replace = Product_Price )
df_scopy['Unit Product Price'].value_counts()
bins=[14,24,40,64]       
label1=['14-24','25-40','41-64']
label2=["Youth", "Young Adults","Old Adults"]
df_scopy['Age Groups']=pd.cut(df_scopy['Age'],bins,labels = label1)
df_scopy['Age Category']=pd.cut(df_scopy['Age'],bins,labels = label2)
df_scopy.shape


# In[788]:


# Changing datatype of Unit Product Price from Object to int64.
df['Unit Product Price'] = df['Unit Product Price'].astype("int64")
df.info()


# In[789]:


#EDA on Univariate Categorical variables
def cat_feat(col_data):
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(8,5))
    fig.suptitle(col_data.name+' Wise Sale',fontsize=15)
    sns.countplot(col_data,ax=ax[0])
    col_data.value_counts().plot.pie(autopct='%1.1f%%',ax=ax[1])
    plt.tight_layout()


# **1.Product**

# In[790]:


print(df.Product.value_counts())


# In[791]:


cat_feat(df['Product'])


# **2.Gender**

# In[792]:


print(df.Gender.value_counts())


# In[793]:


cat_feat(df['Gender'])


# **3.MaritalStatus**

# In[794]:


print(df.MaritalStatus.value_counts())


# In[795]:


cat_feat(df['MaritalStatus'])


# **Observations:**
# - 1.**Derived Category columns** are Unit Product Price, Age Groups, Age Category
# - 2. Product KP281 is the most selling model
# - 3. There are more male buyers then female buyers.
# - 4. Couples are buying more treadmills then singles.
# 

# # EDA - Bivariate Analysis

# In[796]:


# Original dataframe before outliers removal
sns.lineplot(x='Age',y='Income',  data=df,  hue='Product')
plt.show()


# In[797]:


# Modified dataframe after outliers removal
sns.lineplot(x='Age',y='Income',  data=df_scopy,  hue='Product')
plt.show()


# **Observations**:
# - Here we can clearly see that, most of the buyers who have income greater than 80K, prefers to buy product KP781 with advanced features. 
# - Also, as the second graph without income ouliers, we aren't getting any significant disturbances expect the higher income group, hence it's benefetial to keep outliers i.e first (df) for further inferences.

# In[798]:


sns.barplot(x='Age Groups',  y='Income',hue='Product',  data=df)
plt.show()


# In[799]:


sns.countplot(x='MaritalStatus',
    hue='Unit Product Price',
    data=df)
plt.show()
#CONC


# In[800]:


sns.countplot(x='Gender',
    hue='Unit Product Price',
    data=df)
plt.show()
#CONC


# In[801]:


sns.countplot(x='Usage',
    hue='Product',
    data=df)
plt.show()
#CONC


# In[802]:


sns.countplot(x='Usage',
    hue='Gender',
    data=df)
plt.show()
#CONC


# **Observations and Inferences**:
# - From above countplot for Usage , we can clearly see that, as the no. of usage per week of a customer increases (goes beyond 3), then only there's a demand of treadmill with advanced features and highest cost(KP781-> USD 2500) which implies that if a customer is serious and is regular in running, then only he/she prefer purchasing advanced tredmill
# - As the seriousness / regularity in terms of usage per week of the customer increases, they prefers treadmill with advanced features rather than low and middle range product. Which implies, Aerofit, should focus selling more advance range products to the serious customers i.e target audience should be (gym freaks, health coaches, yoga coaches, fitness enthusiast, etc)

# In[803]:


sns.boxplot(x='Usage',
            y = 'Income',
    hue='Product',
    data=df_old)
plt.show()
#CONC


# In[804]:


sns.boxplot(x='Usage',
            y = 'Miles',
    hue='Product',
    data=df_old)
plt.show()
#CONC


# In[805]:


sns.boxplot(x='Fitness',
            y = 'Age',
    hue='Product',
    data=df_old)
plt.show()
#CONC


# In[806]:


pd.crosstab(df['Education'] ,df['Product']).plot(kind= 'bar')
plt.show()


# **Inferences**:
# - **The sales team should focus the high range product's marketing to males who are married and have higher income than 50k and who uses the product more than or equal to 4 times in a week and who have education more than or equal to 16 years**(This should be the target audience for KP781)

# # Creating customer Profile using conditional and marginal probabilities

# In[807]:


df.groupby(by='Product')['Age'].mean() ##Average age of buying product models


# In[808]:


df.groupby('Product')['Income'].mean() ##Average income of buying each model


# In[809]:


print(df[['Product','Gender']].value_counts().sort_index()) ## models bought by different Genders


# **MARGINAL PROBABILITES**

# 1. MARGINAL PROBABILITIES of the customers who are either female or male buying any of the three products:

# In[810]:


pd.crosstab(index=df['Gender'],columns=df['Product'],margins=True)


# In[811]:


marg_prob1 = round(pd.crosstab(index=df['Gender'],columns=df['Product'],margins=True,normalize=True)*100,2)
marg_prob1


# In[812]:


sns.countplot(x='Product',hue='Gender',data=df)
plt.show()


# 2. MARGINAL PROBABILITIES of the customers who usages (from twice a week to 7 times a week) buying any of the three products:

# In[813]:


marg_prob2 = round(pd.crosstab(index=df['Usage'],columns=df['Product'],margins=True,normalize=True)*100,2)
marg_prob2


# In[814]:


sns.countplot(x='Product',hue='Usage',data=df)
plt.show()


# **Observations**:
# - High cost/advanced featured KP781 product usage is more among people who are buying it. So, it's a win - win situation for the company to focus on the target audience - (to **MALES** who are **MARRIED** and have **higher income than 50k** and who uses the product more than or equal to **4 times in a week(usage)** and who have **education more than or equal to 16 years**)

# 3. MARGINAL PROBABILITIES of the customers who are in the age groups(15-64) buying any of the three products:

# In[815]:


marg_prob3 = round(pd.crosstab(index=df['Age Groups'],columns=df['Product'],margins=True,normalize=True)*100,2)
marg_prob3


# In[816]:


df['Age Groups'].value_counts().plot(kind = 'pie',autopct='%.2f')
plt.show()


# 4. MARGINAL PROBABILITIES of the customers who are either married or single and buying any of the three products:

# In[817]:


marg_prob4 = round(pd.crosstab(index=df['MaritalStatus'],columns=df['Product'],margins=True,normalize=True)*100,2)
marg_prob4


# In[818]:


sns.countplot(x='Product',hue='MaritalStatus',data=df)
plt.show()


# 5. MARGINAL PROBABILITIES of the customers who having education in years (from 12yrs to 21yrs ) and buying any of the three products:

# In[819]:


marg_prob5 = round(pd.crosstab(index=df['Education'],columns=df['Product'],margins=True,normalize=True)*100,2)
marg_prob5


# In[820]:


sns.countplot(x='Product',hue='Education',data=df)
plt.show()


# 6. MARGINAL PROBABILITIES of the customers who having income in range (29000 - 40000 as Low Income, 40000 - 60000 as Moderate Income,60000- 80000 as High Income,80000 -105000 as Very High Income) and buying any of the three products:

# In[821]:


marg_prob6 = round(pd.crosstab(index=df['Income Groups'],columns=df['Product'],margins=True,normalize=True)*100,2)
marg_prob6


# In[822]:


sns.countplot(x='Product',hue='Income Groups',data=df)
plt.show()


# **CONDITIONAL PROBABILITIES**

# 1. CONDITIONAL PROBABILITIES of the customers who are either female or male buying any of the three products:

# In[823]:


cond_prob1 = pd.crosstab(df['Gender'], df['Product'], margins = True).apply(lambda x : round(x/len(df),2), axis = 1) * 100
cond_prob1


# 2.CONDITIONAL PROBABILITIES of the customers who usages (from twice a week to 7 times a week) buying any of the three products:

# In[824]:


cond_prob2 = pd.crosstab(df['Usage'], df['Product'], margins = True).apply(lambda x : round(x/len(df),2), axis = 1) * 100
cond_prob2


# 3. CONDITIONAL PROBABILITIES of the customers who are in the age groups(15-64) buying any of the three products:

# In[825]:


cond_prob3 = pd.crosstab(df['Age Groups'], df['Product'], margins = True).apply(lambda x : round(x/len(df),2), axis = 1) * 100
cond_prob3


# 4.CONDITIONAL PROBABILITIES of the customers who are either married or single and buying any of the three products:

# In[826]:


cond_prob4 = pd.crosstab(df['Age Groups'], df['Product'], margins = True).apply(lambda x : round(x/len(df),2), axis = 1) * 100
cond_prob4


# 5. CONDITIONAL PROBABILITIES of the customers who having education in years (from 12yrs to 21yrs ) and buying any of the three products:

# In[827]:


cond_prob5 = pd.crosstab(df['Education'], df['Product'], margins = True).apply(lambda x : round(x/len(df),2), axis = 1) * 100
cond_prob5


# 6. CONDITIONAL PROBABILITIES of the customers who having income in range (29000 - 40000 as Low Income, 40000 - 60000 as Moderate Income,60000- 80000 as High Income,80000 -105000 as Very High Income) and buying any of the three products:

# In[828]:


cond_prob6 = pd.crosstab(df['Income Groups'], df['Product'], margins = True).apply(lambda x : round(x/len(df),2), axis = 1) * 100
cond_prob6


# **Observations and Inferences:**
# - From the customer profiling using marginal probabilty and conditional probability, we can easily get all the stats in percentage - like we can say that there are no one in the very high income group who is willing to purchase KP781.

# # Checking correlation among different features

# In[829]:


df.corr()


# In[830]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,linewidths=0.3, linecolor='white')
plt.show()


# **Inferences**:
# 
# - Age,Education,Usage,Fitness & Miles has significant correlation with Income and vice versa.
# - Usage and Fitness are highly correlated with Miles and vice versa.

# In[831]:


#Quick overview of the data

sns.set_style('white')
sns.pairplot(df,hue='Product')
plt.show()


# **Observations**:
# - KP281 model is the most purchased model (44.4%) then KP481 (33.3%). 
# - KP781 is the least sold model (22.2%).
# - There are more Male customers (57.8%) than Female customers (42.2%).
# - Average Usage of Males is more than Average usage of Females.
# - Customers buying treadmill are younger and average age of customer is 28.
# - Most of the customers earns less than 70K and prefer KP281 & KP481 models.
# - 59.4% of the customers who purchased treadmill are partnered.
# - Customers average education is 16.
# 

# # Multivariate Analysis

# In[832]:


sns.catplot(x='Usage', y='Income', col='Gender',hue='Product' ,kind="bar", data=df) 
plt.show()


# - Customers having lower income range (<60K) prefer to buy models KP281 & KP481 and expect to use treadmill 2-5 times/week.
# - Mostly Higher earning customers bought KP781 and expect to use treadmill 4-6 times/week.

# In[833]:


sns.catplot(x='Gender',y='Income', hue='Product', col='MaritalStatus', data=df,kind='bar')
plt.show()


# In[834]:


pd.crosstab(index=df['Product'], columns=[df['MaritalStatus'],df['Gender']] )  


# - Partnered Female bought KP281 Model compared to Partnered male.
# - Partnered Male customers bought KP481 & KP781 models more than Single Male customers.
# - Single Female customers bought KP481 model more than Single male customers.
# - Single Male customers bought KP281 & KP781 models compared to Single females.
# - The majority of treadmill buyers are men.

# In[835]:


#Some information on how many miles are run per week or per session: 
df['Miles per session'] = df['Miles']/df['Usage']
sns.histplot(x='Miles per session', data=df, hue = 'Product')
plt.axvline(np.mean(df['Miles per session']),color='r',linestyle='--')
plt.xlabel('Miles per session')
plt.show()


# - KP481 is used for longer sessions
# - KP281 is used for shorter or moderate sessions

# In[836]:


sns.histplot(x='Miles',data=df,hue='Product',multiple='dodge')
plt.axvline(np.mean(df['Miles']),color='r',linestyle='--')
plt.xlabel('Miles per week')
plt.show()


# In[837]:


fig, ax = plt.subplots(figsize=[10,5])
sns.histplot(x='Product',data=df,hue='Fitness',alpha=0.5, element='bars',stat='density',multiple='dodge')
plt.ylabel('Normalized count')
plt.show()


# - Education level is directly correlated with income as highlighted in the pairplot and correlation heatmap above, so highly educated indviduals are more likely to purchase the more expensive model 

# In[838]:


sns.stripplot(x='Product',y='Education',data=df)
plt.show()


# In[839]:


sns.stripplot(x='Product',y='Income',data=df)
plt.show()


# # Final Observations and Inferences

# - Total sales for Aerofit treadmills of unit price USD 1500(KP281) is USD 120000, USD 1750(KP481) is USD 105000, USD 2500(KP781) is USD 100000
# 
# - KP781 Treadmill with advanced features is preffered by the customers with higher income.
# 
# - KP281 Treadmill with the lowest cost and basic features is preffered by the customers with lower income and the KP481 product with moderate features are liked by the customers with upper bracket of low - moderate income group.
# 
# - KP281 model is the most purchased model (44.4%) then KP481 (33.3%). 
# - KP781 is the least sold model (22.2%).
# - There are more Male customers (57.8%) than Female customers (42.2%).
# - Average Usage of Males is more than Average usage of Females.
# - Customers buying treadmill are younger and average age of customer is 28.
# - Most of the customers earns less than 70K and prefer KP281 & KP481 models.
# - 59.4% of the customers who purchased treadmill are partnered.
# - Customers average education is 16.
# - Most customers have income less than 70k.
# - Customers run on an average 80 miles per week.
# 
# - There aren't any significant outliers for Bivariate Analysis of Products and the Income of customers purchasing those products. So no need for outlier removal here.
# 
# - After outlier removal for income, 19 rows are deleted and in order to draw some insights from the original data in future, stored the modied data in it's shallow copy - **df_scopy**
# 
# - In the boxplot, we can clearly see that most of the outliers are removed and the data is now ready for further analysis and inferences.
# 
# - After dealing with outliers, we can infer that, there's a 7.31% correction in the Income data after removing outliers as we can see that the difference in the mean and median has decreased from 3123 to 228. Hence the new dataframe i.e **df_scopy**is more suitable for carrying further analysis w.r.t income and gender related cases.
# 
# - Most of the buyers who have income greater than 80K, prefers to buy product KP781 with advanced features.
# 
# - **Significant no. of Outliers (higher end) are present** as there are very few persons who earn >80k. This makes us mandatory to handle outliers which has been taken care in the first case. Shallow copy of our dataframe(**ds_scopy**) consists of modified data after dealing with ouliers.
# 
# - Also, After further analysis, we got to know that, as the **with dataframe without income ouliers, we aren't getting any significant disturbances expect the higher income group, hence it's benefetial to keep outliers i.e first (df) for further inferences as if we use df_scopy, then it might lead us to falsification of data due to data deletion**
# 
# - Customers having lower income range (<60K) prefer to buy models KP281 & KP481 and expect to use treadmill 2-5 times/week.
# 
# - Mostly Higher earning customers bought KP781 and expect to use treadmill 4-6 times/week.
# 
# 
# **Inferences with Customer Profiles** : 
# 
# 
# - The target audience for KP781 Treadmill should be the higher income group. So the sales team must focus on this range.
# 
# - **The sales team should focus the high range product's marketing to males who are married and have higher income than 50k and who uses the product more than or equal to 4 times in a week and who have education more than or equal to 16 years**(This should be the target audience for KP781)
# 
# - High cost/advanced featured KP781 product usage is more among people who are buying it. So, it's a win - win situation for the company to focus on the target audience - (to **MALES** who are **MARRIED** and have **higher income than 50k** and who uses the product more than or equal to **4 times in a week(usage)** and who have **education more than or equal to 16 years**)
# 
# - Education level is directly correlated with income as highlighted in the pairplot and correlation heatmap above, so highly educated indviduals are more likely to purchase the more expensive model.The sales team should focus on this aspect.
# 
# 
# 
# **1. KP281**
# 
# - Customers who bought this treadmill have income less than 60k with an average of 55K.
# - This model has same level of popularity in Male customers as well as Female customers as it has same numbers of Male and Female customers.
# - Average age of customer who purchases KP281 is 28.5.
# - This model is popular among Bachelors as average years of education of customers for this product is 15.
# - Self rate fitness level of customer is average.
# - Customers expect to use this treadmill 3-4 times a week.
# - It is the most popular model (in all genders) because of its appealing price and affordability with 33.3% of sales.
# - Customers who bought this treadmill want fitness level atleast average and maybe they were looking for a basic treadmill with appealing price that also does the job.
# 
# 
# **2. KP481**
# 
# - This model is second most sold model with 33.3% of sales.
# - Customers with lower income purchase KP281 and KP481 model may be because of lower cost of the Treadmill.
# - Average age of customer who purchases KP481 is 29.
# - This model is popular among Bachelors as average years of education of customers for this product is 16.
# - Customers expecting KP481 model to use less frequently but to run more miles per week on this.
# - This model is popular more in Single Female customers compare to Single male customers may be because of difference in provided features or color scheme.
# 
# **3. KP781**
# 
# - This is the least sold product(22.2% sales) in company lineup of Treadmill may be because of it heafty price range making it Company's Premium product.
# - This model is popular with customers having high income range as average Income is 75K .
# - Average age of customer who purchases KP781 is 29.
# - This model is popular among Customers with higher education as average education is 17 years.
# - Treadmill may have some advanced features as people with high income are ready to spend money to buy this model
# - Customers expected usage on this model is 4-5 day a week with moderate Miles to run having average 166 miles per week.
# - Male customers who are more serious about fitness or Professionals buy this mode (self fitness rating 3-5).
# - From the customer profiling using marginal probabilty and conditional probability, we can easily get all the stats in percentage - like we can say that there are no one in the very high income group who is willing to purchase KP781.

# **To conclude, we can get complete profile report by using Pandas inbuilt function called ProfileReport**

# In[840]:


ProfileReport(df)

