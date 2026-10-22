from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
OUT=BASE/"outputs"; OUT.mkdir(exist_ok=True)
df=pd.read_csv(BASE/"logistics_prediction_dataset.csv")
num=["Distance_km","Shipment_Volume_units","Month","Carrier_Experience_years","Transport_Cost_INR"]
cat=["Transport_Mode","Destination_Type","Weather","Traffic_Level","Weekend_Delivery"]
X=df[num+cat]; y=df["Transit_Time_days"]
prep=ColumnTransformer([
 ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
 ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
rows=[]
for name,model in [("Linear Regression",LinearRegression()),("Random Forest baseline",RandomForestRegressor(n_estimators=200,min_samples_leaf=2,random_state=42))]:
 p=Pipeline([("prep",prep),("model",model)]); p.fit(Xtr,ytr); yhat=p.predict(Xte)
 rows.append({"Model":name,"MAE_days":mean_absolute_error(yte,yhat),"RMSE_days":np.sqrt(mean_squared_error(yte,yhat)),"R2":r2_score(yte,yhat)})
cv=KFold(n_splits=5,shuffle=True,random_state=42)
search=GridSearchCV(Pipeline([("prep",prep),("model",RandomForestRegressor(random_state=42))]),
 {"model__n_estimators":[100,200],"model__max_depth":[None,10],"model__min_samples_leaf":[1,2]},
 scoring="neg_mean_absolute_error",cv=cv,n_jobs=-1)
search.fit(Xtr,ytr); model=search.best_estimator_; yhat=model.predict(Xte)
rows.append({"Model":"Tuned Random Forest","MAE_days":mean_absolute_error(yte,yhat),"RMSE_days":np.sqrt(mean_squared_error(yte,yhat)),"R2":r2_score(yte,yhat)})
pd.DataFrame(rows).to_csv(OUT/"model_metrics.csv",index=False)
scores=-cross_val_score(model,Xtr,ytr,scoring="neg_mean_absolute_error",cv=cv,n_jobs=-1)
(OUT/"validation_summary.txt").write_text(f"Best parameters: {search.best_params_}\\nMean 5-fold CV MAE: {scores.mean():.3f} days\\nCV MAE standard deviation: {scores.std():.3f} days\\n")
pd.DataFrame({"Actual":yte,"Predicted":yhat},index=yte.index).to_csv(OUT/"test_predictions.csv")
pi=permutation_importance(model,Xte,yte,n_repeats=8,random_state=42,scoring="neg_mean_absolute_error")
pd.Series(pi.importances_mean,index=Xte.columns).sort_values(ascending=False).to_csv(OUT/"feature_importance.csv")
plt.figure(figsize=(7,5)); plt.scatter(yte,yhat,alpha=.7); lo=min(yte.min(),yhat.min()); hi=max(yte.max(),yhat.max()); plt.plot([lo,hi],[lo,hi],"--"); plt.xlabel("Actual days"); plt.ylabel("Predicted days"); plt.title("Actual vs predicted"); plt.tight_layout(); plt.savefig(OUT/"actual_vs_predicted.png",dpi=150); plt.close()
print(pd.DataFrame(rows).round(3).to_string(index=False)); print("Best parameters:",search.best_params_); print("CV MAE:",round(scores.mean(),3))
