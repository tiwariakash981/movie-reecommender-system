#project ka pura flow bana ki kaise karna hai aur github pe uploa karna 

import pandas as pd
import numpy as np
import ast
from ast import literal_eval 
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer#used for removing similar words and making them one word
from sklearn.metrics.pairwise import cosine_similarity#ab har ek vector ka apas me kitna distance hsi ye check karenge 
import pickle


movie = pd.read_csv(r'C:\AKASH\Projects\MOVIE RECOMMENDER SYSTEM\datasets/tmdb_5000_movies.csv')
credit = pd.read_csv(r'C:\AKASH\Projects\MOVIE RECOMMENDER SYSTEM\datasets/tmdb_5000_credits.csv')
print(movie.head())

print(movie.shape,credit.shape,'ak')

movies = movie.merge(credit,on='title')
print(movies.shape)
print(movies.info())

# Set pandas options to display all columns and rows
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)


#columns which we will use
#genre,id,keywords,title,overview,cast,crew
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
print(movies.head(1))

movies.dropna(inplace=True)
print(movies.isnull().sum())
print(movies.duplicated().sum())

#abhi genre, keywords, overview, cast, crew ye sab ko string format me daal ke ek tags
#naam ke column me daal denge taki tag ke andar sara details ek hi baar me miml jaye

print(movies.iloc[0].genres)#mujhe sabka 'name' chahiye isliye ek function bananunga aur return kar dunga

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
         L.append(i['name'])
    return L
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
#don't open this comment##convert(({"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}))
print(movies.head())

print(movies['cast'][0])#cast = jo movie me kaam kiye hai , isme main character 
def convert3(obj):
    counter=0
    L = []
    for i in ast.literal_eval(obj):
        if counter < 3:
            L.append(i['name'])
            counter+=1
    return L
movies['cast'] = movies['cast'].apply(convert3)
print(movies.head())

print(movies['cast'][0])#ye kiya hu taki check karke pata chale ki barobar se teen value mila hai ki nhi teen hi value liya kyuki vo log most famous character the isme 

print(movies['crew'][0])#ye kiya hai check karne ke liye ki konsa field apne kaam aaega here job:'Director'

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L
            
movies['crew'] = movies['crew'].apply(fetch_director)
print(movies['crew'])

movies['overview'] = movies['overview'].apply(lambda x:x.split())
print(movies.head(2))

#now main kaam hogaya lekin abhi Sam Worthington hai to usko ek single naam likhna padega nhi to machine ko lagega ki dono alag hai

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(' ','') for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(' ','') for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(' ','') for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(' ','') for i in x])

movies['tags'] = movies['overview'] + movies['crew'] + movies['cast'] + movies['keywords']
##print(movies.head(2))

new_df = movies[['movie_id','title','tags']]
print(new_df)

#ab tags me sabko lowercase aur join kar denge thoda data samjhega apne ko 
new_df['tags'] = new_df['tags'].apply(lambda x: ' '.join(x))

print(new_df['tags'][0])

new_df['tags'].apply(lambda x:x.lower())
print(new_df['tags'][0])

#now doing vectorization i.e 2d space me line draw karunga taki ek movie ka naam mile to phir 
#countvectorizer automatically gin leta hai ki ek word kitne baar me aa rha hai ek paragraph me

cv = CountVectorizer(max_features = 5000 , stop_words = 'english')
vectors = cv.fit_transform(new_df['tags']).toarray()
print('all vectors on zeroth position',vectors[0])
print('shape of vectors',vectors.shape)#yaha humlog 5000 features lene bola hai 
print(len(cv.get_feature_names_out()))#ye vo 5000 words hai jo zada baar apne string me aaye hai 

#ab mujhe activity aur acitivities ye same jaise word ko ek karna hai
#isliye humlog stemming technique ka use karenge

ps = PorterStemmer()
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return ''.join(y)#jo list me mila tha usko vapis string me convert kiya hai
print(ps.stem('loved'))#example showing 

new_df['tags'] = new_df['tags'].apply(stem)

#ab main kaam hai ki similar movies provide karna i.e ab aoun jo vectorbanaye hai un sabka cosine(angle distance) distance calculate karenge
#ek eucledian distance hota hai jaise ki do line ke beech ka distance from its tip formula:(x2-x1)^2 + (y2-y1)^2 ....
#high dimension use kiya hai isliye cosine distance hi use karna hai 

similarity = cosine_similarity(vectors)#vectors ko upar define kiya hai  i.e isme vector form kar rahe hai apne data ke tags column ko 
print(similarity[0])#0th movie ka baki sabse similarity 

#main function = har ek movie ka aapas me similarity nikalke top ke 5 movies ko recommend karna hai
#

##sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
##        print(i[0])
        print(new_df.iloc[i[0]].title)

recommend('Batman Begins')#checking ki kaam kar rha hai ki nahi

##pickle.dump(new_df,open('movies.pkl','wb'))

pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))

pickle.dump(similarity,open('similarit.pkl','wb'))


























