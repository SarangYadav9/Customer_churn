print('-------------------------------------Customer Churn Prediction--------------------------------------')
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/dexter/Documents/python/customer_churn.csv')
df['Churn']=df['Churn'].map({'No':0,'Yes':1})
df['Contract']=df['Contract'].map({'Month-to-month':0,'One year':1,'Two year':2})

df.dropna(inplace=True)

features = ['tenure','Contract','MonthlyCharges']
x = df[features]
y = df['Churn']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)

tenure = int(input("Enter tenure :"))
contract = int(input("Enter contact(0->Month-to-Month,1->One year,2->Two years) :"))
monthlycharges = float(input("Enter monthly charges :"))
df1 = pd.DataFrame([[tenure,contract,monthlycharges]],columns=features)
pred = model.predict(df1)

if pred[0]==1:
    print("The customer will churn.")
else:
    print("The customer will not churn.")
print("Program ended Thankyou!")