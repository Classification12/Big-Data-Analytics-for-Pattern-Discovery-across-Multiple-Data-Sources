
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def Model_Kmeans(X_train, X_test, y_train, y_test, Act):
    # ---- 2. Apply K-Means Clustering ----
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    y_clusters = kmeans.fit_predict(X_test)  # Use predicted clusters as labels
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # ---- 4. Evaluate Classification Model ----
    y_pred = y_clusters.predict(X_test)
    Eval = clf.predict(y_pred, y_test)
    return y_pred, Eval
