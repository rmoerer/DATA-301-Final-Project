# %% [markdown]
# ## Classifying Songs into Happy and Sad
# 
# #### By Ryan Moerer

# %%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from secret import client_id, client_secret # import my Spotify API credentials
np.random.seed(42)

# %% [markdown]
# ## Data Collection

# %%
# Spotify API authorization
credentials = SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret
)

sp = spotipy.Spotify(auth_manager=credentials)

# %%
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def get_tracks(query):
    # get 50 playlists for query
    playlist_res = sp.search(query, type='playlist', limit=50)
    playlists = {}
    for playlist in playlist_res['playlists']['items']:
        playlists[playlist['name']] = playlist['id']
    # get all tracks into df
    all_tracks = []
    for name, id in playlists.items():
        ofst = 0
        while True:
            response = sp.playlist_tracks(id, offset=ofst, limit=100)
            for track in response['items']:
                if track['track']:
                    all_tracks.append({'playlist_name': name,'playlist_id': id,'name': track['track']['name'],'album_name': track['track']['album']['name'],
                        'artist': track['track']['artists'][0]['name'],'id': track['track']['id'],'popularity': track['track']['popularity'],'explicit': track['track']['explicit']
                    })
            ofst += 100
            if not response['next']:
                break
    tracks_df = pd.DataFrame(all_tracks).drop_duplicates(subset=['id'])
    # get all features into df
    features = []
    for chunk in chunker(tracks_df['id'].tolist(), 100):
        features.extend(sp.audio_features([id for id in chunk if id]))
    features_df = pd.DataFrame([feature for feature in features if feature]).drop(columns=["uri","track_href","analysis_url","type"])
    return pd.merge(tracks_df, features_df, left_on="id", right_on="id", how="inner")

# %%
# create dataframe
# commented out as to not accidentally overwrite dataset
#happy_df = get_tracks("happy songs")
#sad_df = get_tracks("sad songs")
#happy_df['y'] = 1
#sad_df['y'] = 0
#data = pd.concat([happy_df, sad_df])
#data = data.groupby('id').filter(lambda x: len(x)<2) # in case some songs are classified as happy and sad
#g = data.groupby('y')
#data = g.apply(lambda x: x.sample(g.size().min(), random_state=3)).reset_index(drop=True)
#data.to_csv("songs_data.csv")
#data

# %% [markdown]
# ## Data Cleaning and Preprocessing

# %%
data = pd.read_csv("songs_data.csv", index_col=0)
data

# %%
# create training and testing sets
X = data.iloc[:,7:-1].drop(columns=['time_signature'])
X = pd.get_dummies(X, columns=['key'])
y = data['y']
X_features = X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, stratify=y, random_state=12)

# Standardize training and testing sets
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# %%
# Example of cleaned data
X_train.iloc[0]

# %% [markdown]
# ## Visualizing the Data

# %%
# Create correlation matrix plot
cols = ['key_' + str(i) for i in range(0,12)]
cols + ['mode','explicity']
corr_matrix = X_train.drop(columns=cols).corr()
heatmap_plot = sns.heatmap(corr_matrix, cmap="Blues", annot=True, fmt=".2f")
fig = heatmap_plot.get_figure()
fig.set_size_inches(12, 10)
fig.savefig("corr_mat.png")

# %%
# PCA and PCA plot
pca = PCA(n_components=3, random_state=30)
pca_values = pca.fit_transform(X_train_scaled)

fig = plt.figure()
ax = plt.axes(projection='3d')

x = pca_values[:,0]
y = pca_values[:,1]
z = pca_values[:,2]

scatter1_proxy = plt.Line2D([0],[0], linestyle="none", marker = 'o', color='#F9E721')
scatter2_proxy = plt.Line2D([0],[0], linestyle="none", marker = 'o', color='#450256')

colors = {1: '#F9E721', 0: '#450256'}

ax.scatter(x,y,z, c = y_train.map(colors))
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
ax.legend([scatter1_proxy, scatter2_proxy], ['Happy', 'Sad'], numpoints = 1)
fig.set_size_inches(10,10)
plt.savefig("pca.png")

# %% [markdown]
# ## KNN

# %%
# standardize the training features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# create param grid
params = {'n_neighbors': list(range(1,51))}

# KNN classifier with Minkowski distance
knn = KNeighborsClassifier(weights='distance')

# grid search
knn_gridsearch = GridSearchCV(knn, param_grid=params, scoring='accuracy', cv=10)
knn_gridsearch.fit(X_train_scaled, y_train)

# %%
# best training accuracy and params
knn_gridsearch.best_params_, knn_gridsearch.best_score_

# %% [markdown]
# ## Logistic Regression

# %%
10**np.linspace(-4,5,20)

# %%
# create param grid
params = {'C':10**np.linspace(-4,5,20)}

# Logistic Regression with l2 regularization
log_reg = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)

# grid search
log_gridsearch = GridSearchCV(log_reg, params, scoring='accuracy', cv=10)
log_gridsearch.fit(X_train_scaled, y_train)

# best params/best cross-validated accuracy
log_gridsearch.best_params_, log_gridsearch.best_score_

# %%
pd.Series(log_gridsearch.best_estimator_.coef_[0], index=X_features)

# %% [markdown]
# ## Random Forests

# %%
# initialize RF model
rf_clf = RandomForestClassifier(random_state=42)

# fit model on training (not using standardized data for RF)
rf_clf.fit(X_train, y_train)

# %%
# create feature importance plot
feat_importances = rf_clf.feature_importances_
indices = np.argsort(feat_importances)

fig = plt.figure()
plt.barh(range(len(feat_importances)), feat_importances[indices])
plt.yticks(range(len(feat_importances)), labels=np.array(X_features)[indices])
plt.xlabel("Feature Importance")
fig.set_size_inches(12,12)
plt.savefig("feature_importance.png")
plt.show()

# %% [markdown]
# ## Evaluating the Models

# %%
rf_preds = rf_clf.predict(X_test)
log_preds = log_gridsearch.predict(X_test_scaled)
knn_preds = knn_gridsearch.predict(X_test_scaled)

pd.DataFrame({
    'Precision': [precision_score(y_test, rf_preds),precision_score(y_test,log_preds),precision_score(y_test,knn_preds)],
    'Recall': [recall_score(y_test, rf_preds),recall_score(y_test,log_preds),recall_score(y_test,knn_preds)],
    'F1 Score': [f1_score(y_test, rf_preds),f1_score(y_test,log_preds),f1_score(y_test,knn_preds)],
    'Accuracy': [accuracy_score(y_test, rf_preds),accuracy_score(y_test,log_preds),accuracy_score(y_test,knn_preds)],
    }, 
    index=['Random Forest', 'KNN', 'LogisticRegression']).sort_values(by='Accuracy', ascending=False)


