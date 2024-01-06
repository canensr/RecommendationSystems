#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################

# !pip install surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Modelleme
# Adım 3: Model Tuning
# Adım 4: Final Model ve Tahmin

#############################
# Adım 1: Veri Setinin Hazırlanması
#############################

movie = pd.read_csv('Recommendation_Systems/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('Recommendation_Systems/datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
df.head()

movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

sample_df = df[df.movieId.isin(movie_ids)]
sample_df.head()

sample_df.shape

user_movie_df = sample_df.pivot_table(index=["userId"],
                                      columns=["title"],
                                      values="rating")

user_movie_df.shape

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(sample_df[['userId',
                                       'movieId',
                                       'rating']], reader)

##############################
# Adım 2: Modelleme
##############################

trainset, testset = train_test_split(data, test_size=.25)
svd_model = SVD()
svd_model.fit(trainset) #trainset üzerinden öğren fit et.
predictions = svd_model.test(testset) #testset in değerlerini tahmin etmeye çalışıcak

accuracy.rmse(predictions) # hata kareler ortalmasının karekökü
                           # tahmin etmede beklenen ortalama hatadır


svd_model.predict(uid=1.0, iid=541, verbose=True)

svd_model.predict(uid=1.0, iid=356, verbose=True)


sample_df[sample_df["userId"] == 1]

##############################
# Adım 3: Model Tuning
##############################

param_grid = {'n_epochs': [5, 10, 20],      #dışsal parametredir kullanıcı tarafından verilmesi gerekir
              'lr_all': [0.002, 0.005, 0.007]}


gs = GridSearchCV(SVD,
                  param_grid,
                  measures=['rmse', 'mae'],
                  cv=3,
                  n_jobs=-1,
                  joblib_verbose=True)

gs.fit(data)

gs.best_score['rmse'] #en iyi hata kareler ortlaması
gs.best_params['rmse'] #sonucu veren en iyi parametreler nelerdir onları göstercek

#tanımlı değerle optime çıkan değer farklı olduğu için yeniden model kurup
# optime değeri modele vermemeiz gerekecek.

##############################
# Adım 4: Final Model ve Tahmin
##############################

dir(svd_model)
svd_model.n_epochs

svd_model = SVD(**gs.best_params['rmse'])

data = data.build_full_trainset()
svd_model.fit(data) #bütün modeli kullanarak büütn verileri kullanarak modeli tekrardan fit ediyoruz.

svd_model.predict(uid=1.0, iid=541, verbose=True)






