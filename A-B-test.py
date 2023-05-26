#!/usr/bin/env python
# coding: utf-8

# # 11

# ## Общее описание:
# Есть данные о такси-компании (uber), которая хочет изучить отток водителей и посмотреть, какие есть различия между водителями, которые покидают сервис, и которые остаются. Нужно сформулировать и протестировать гипотезы, выделить группы водителей, которые наиболее подвержены "оттоку". На основе результатов сделать выводы о том, что можно улучшить в сервисе, чтобы в дальнейшем внести изменения (и после – провести A/B тест и выяснить, стало ли лучше).
# 
# ### Описание данных
# 
# - `city` – город
# - `phone` – основное устройство, которое использует водитель 
# - `signup_date` – дата регистрации аккаунта (`YYYYMMDD`)
# - `last_trip_date` – дата последней поездки (`YYYYMMDD`)
# - `avg_dist` – среднее расстояние (в милях) за поездку в первые 30 дней после регистрации
# - `avg_rating_by_driver` – средняя оценка поездок водителем 
# - `avg_rating_of_driver` – средняя оценка поездок водителя
# - `surge_pct` – процент поездок, совершенных с множителем > 1 (кажется когда большая загруженность и тд)
# - `avg_surge` – средний множитель всплеска за все поездки этого водителя
# - `trips_in_first_30_days` – количество поездок, которые совершил водитель в первые 30 дней после регистрации
# - `luxury_car_user` – TRUE, если пользователь в первые 30 дней использовал премиум-автомобиль
# - `weekday_pct` – процент поездок пользователя, совершенных в будние дни
# 
# 
# ### План
# 
# 1. **Сначала сделаем небольшой препроцессинг:**
#     - Посмотрим на данные
# 2. **Далее сформулируем гипотезы, исходя из общей задачи:**
#     - Сформулируем предположения, которые будем тестировать
#     - Создадим лейбл churn/not_churn
#     - Построим графики
#     - **Поинт:** только по графикам выводы делать – bad practice, хорошо подкреплять стат. тестами (и стат. тесты есть не только в A/B)
# 3. **Тестируем гипотезы:**
#      - Выбираем гипотезу
#      - Выбираем подходящий тест
#      - Тестируем
# 4. **Подводим итоги:**
#     - Сформулировать выводы и суммаризировать всё что было
#     - Какие действия нужно предпринять разработчикам/бизнесу, чтобы стало лучше? Как можно будет позже провести A/B тестирование? (починить android приложение, возможно таргетить и мотивировать не очень активных водителей, улучшить программу лояльности и бонусов для водителей и тд и тп)
# 
# 

# ## 1: загружаем
# Еще раз список переменных:
# 
# - `city` – город
# - `phone` – основное устройство, которое использует водитель 
# - `signup_date` – дата регистрации аккаунта (`YYYYMMDD`)
# - `last_trip_date` – дата последней поездки (`YYYYMMDD`)
# - `avg_dist` – среднее расстояние (в милях) за поездку в первые 30 дней после регистрации
# - `avg_rating_by_driver` – средняя оценка поездок водителем 
# - `avg_rating_of_driver` – средняя оценка поездок водителя
# - `surge_pct` – процент поездок, совершенных с множителем > 1 
# - `avg_surge` – средний множитель всплеска за все поездки этого водителя
# - `trips_in_first_30_days` – количество поездок, которые совершил водитель в первые 30 дней после регистрации
# - `luxury_car_user` – TRUE, если пользователь в первые 30 дней использовал премиум-автомобиль
# - `weekday_pct` – процент поездок пользователя, совершенных в будние дни
# 

# In[3]:


import numpy as np
import pandas as pd
import scipy.stats as ss

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(12,6)}, style="whitegrid")


# In[4]:


df = pd.read_csv('churn.csv')


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.isna().sum()


# In[8]:


df.nunique()


# In[9]:


df.dtypes


# Изменяем тип для дат:
# 

# In[10]:


df.last_trip_date = pd.to_datetime(df.last_trip_date)
df.signup_date = pd.to_datetime(df.signup_date)


# In[11]:


df.info()


# In[12]:


df.describe()


# In[13]:


df.describe(include='object')


# In[14]:


df.describe(include='datetime')


# In[15]:


df.head()


# ## 2: графики, гипотезы и тесты

# Создаем лейбл churn – пользователь ушел, если не был активен последние 30 дней (но можно попробовать и другие значения в зависимости от вашей компании/данных)

# In[16]:


df.last_trip_date.max()


# In[17]:


df['days_since_last_trip'] = df.last_trip_date.max() - df.last_trip_date


# In[18]:


df['days_since_last_trip']


# Преобразуем в int:

# In[19]:


df['days_since_last_trip'] = df['days_since_last_trip'].dt.days


# In[20]:


df['days_since_last_trip']


# In[21]:


df['churn'] = df.days_since_last_trip.apply(lambda x: 'churn' if x > 30 else 'not_churn')
df[['days_since_last_trip', 'churn']]


# ### churn
# – вы куда все пошли?
# 
# Видим, что очень много пользователей не использовали сервис в последнем месяце. Нужно разобраться, какие факторы могут влиять на отток водителей

# In[22]:


df.churn.value_counts(normalize=True).mul(100)


# In[23]:


fig = px.histogram(df, x='churn')
fig.show()


# Еще лучше – отразим на графике нормализованные значения (сравниваем не сырые числа):

# In[24]:


fig = px.histogram(df, x='churn', histnorm='probability density')
fig.show()


# ### churn & phone
# 
# Предположим, что проблема может быть среди юзеров на конкретной платформе:

# In[25]:


pd.crosstab(df.churn, df.phone)


# In[26]:


fig = px.histogram(df[['churn', 'phone']].dropna(), x='churn', 
                   color='phone')
fig.show()


# Делать вывод только по графику – не очень хорошо, поэтому проверим нашу гипотезу с помощью статистического теста.
# 
# Есть две категориальные переменные → нужен хи-квадрат
# 
# - $H_0$: взаимосвязи между переменными нет 
# - $H_1$: взаимосвязь есть

# In[27]:


from scipy.stats import chi2_contingency, chi2 


# In[28]:


stat, p, dof, expected = chi2_contingency(pd.crosstab(df.churn, df.phone))


# In[29]:


stat, p


# Интерпретируем результат:

# In[30]:


"""prob = 0.95
critical = chi2.ppf(prob, dof)
if abs(stat) >= critical:
    print('Отклоняем H0')
else:
    print('Не отклоняем H0')"""


# In[31]:


prob = 0.95
alpha = 1.0 - prob
if p <= alpha:
    print('Отклоняем H0')
else:
    print('Не отклоняем H0')


# In[32]:


df


# In[38]:


df1=df.query('churn=="churn"').groupby('city').agg({'phone':'count'})
df1


# In[41]:


df2=df.query('churn=="not_churn"').groupby('city').agg({'phone':'count'})
df2


# In[51]:


p


# In[53]:


df3=pd.merge(df1,df2, on=['city'])
df3


# In[54]:


# используем хи-квадрат
from scipy.stats import chi2_contingency
table = df3
stat, p, dof, expected = chi2_contingency(table)
p


# In[55]:


df


# In[56]:


df4=df.query('city=="Astapor"')
df4


# In[57]:


# Проверяем на нормальность 
from scipy import stats
stats.shapiro(df4.trips_in_first_30_days)


# In[58]:


#И строим график для наглядности
import statsmodels.api as sm
import matplotlib.pyplot as plt

fig = sm.qqplot(df4.trips_in_first_30_days)
plt.show()


# In[60]:


df5=df.query('city=="Winterfell"')
df5


# In[61]:


# Проверяем на нормальность 
from scipy import stats
stats.shapiro(df5.trips_in_first_30_days)


# In[64]:


#И строим график для наглядности
import statsmodels.api as sm
import matplotlib.pyplot as plt

fig = sm.qqplot(df5.trips_in_first_30_days)
plt.show()


# In[69]:


xx=np.log(df5.trips_in_first_30_days)
xx


# In[70]:


fig = sm.qqplot(xx)
plt.show()


# In[71]:


stats.shapiro(xx)


# In[72]:


df6=df.query('city!="Winterfell" &city!="Astapor"')


# In[73]:


df6


# In[75]:


#Проводим тест 
stats.kruskal(df4.trips_in_first_30_days, df5.trips_in_first_30_days, df6.trips_in_first_30_days)


# In[76]:





# In[81]:


stats.shapiro(df7.trips_in_first_30_days)


# In[82]:


fig = sm.qqplot(df7.trips_in_first_30_days)
plt.show()


# In[80]:


df7=df.query('churn=="churn"')
df7


# In[83]:


df8=df.query('churn=="not_churn"')
df8.


# In[85]:


#perform the Mann-Whitney U test
stats.mannwhitneyu (df7.trips_in_first_30_days, df8.trips_in_first_30_days)


# In[ ]:


dffin=pd.merge(df1,df2, on=['city'])
dffin


# In[ ]:


# используем хи-квадрат
from scipy.stats import chi2_contingency
table = df3
stat, p, dof, expected = chi2_contingency(table)
p

