from sklearn.decomposition import TruncatedSVD

from Evaluation import evaluation


def Model_FLSA(Train_Data, Train_Target, Test_Data, Test_Target,batch):
    # ---- 4. Apply Latent Semantic Analysis (LSA) using SVD ----
    n_topics = 2  # Number of dimensions (topics)
    lsa = TruncatedSVD(n_components=n_topics, random_state=42)
    X_lsa = lsa.fit_transform(Train_Data)
    Predict = X_lsa.predict(Test_Data)
    Eval = evaluation(Predict, Test_Target)
    return Predict, Eval