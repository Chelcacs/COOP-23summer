from pytorch_tabnet.tab_model import TabNetClassifier
import pandas as pd
from sklearn.metrics import roc_auc_score
# define new model with basic parameters and load state dict weights

loaded_clf = TabNetClassifier()
saved_filepath = "./tabnet_model_test_1.zip"
dataset_name = "DATASET NAME"
loaded_clf.load_model(saved_filepath)

dataobj = pd.read_csv("../data_preprocess/standardized_data.csv")
print(dataobj)
# label_col = "target"
# features = [ col for col in dataobj.columns if col !=[label_col]] 


# X_test = dataobj[features].values[0]
# y_test = dataobj[label_col].values
rows = [0,1,2,3]
x_features = [ col for col in dataobj.columns if col !="target"]
y_label = "target"
X_test = dataobj.loc[rows, x_features].to_numpy()
y_test = dataobj.loc[rows, y_label].to_numpy()


loaded_preds = loaded_clf.predict_proba(X_test)
loaded_test_auc = roc_auc_score(y_score=loaded_preds[:,1], y_true=y_test)

print(f"FINAL TEST SCORE FOR {dataset_name} : {loaded_test_auc}")
