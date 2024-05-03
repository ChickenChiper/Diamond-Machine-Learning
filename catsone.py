#Supressing the warning messages
import warnings
warnings.filterwarnings('ignore')

#Reading the dataset
import pandas as pd
import numpy as np
DiamondData=pd.read_csv('diamonds.csv', encoding='latin')
print('Shape before deleting duplicate values:', DiamondData.shape)

#Removing duplicate rows if any
DiamondData=DiamondData.drop_duplicates()
print('Shape after deleting duplicate values:', DiamondData.shape)

#Printing sample data
#Start observing the Quanntitative/Categorical/Qualitative variables
DiamondData.head(10)

FinalSelections= ['carat', 'cut', 'color', 'clarity']
MLData = DiamondData[FinalSelections]
MLData.head()

MLDataNum=pd.get_dummies(MLData)
MLDataNum['price'] = DiamondData ['price']
MLDataNum.head()

Target_Variable='price'
Predictors=['carat', 'cut_Fair', 'cut_Good', 'cut_Ideal', 'cut_Premium',
       'cut_Very Good', 'color_D', 'color_E', 'color_F', 'color_G', 'color_H',
       'color_I', 'color_J', 'clarity_I1', 'clarity_IF', 'clarity_SI1',
       'clarity_SI2', 'clarity_VS1', 'clarity_VS2', 'clarity_VVS1',
       'clarity_VVS2']

X=MLDataNum[Predictors].values
y=MLDataNum[Target_Variable].values

from tkinter import *
import numpy as np
from xgboost import XGBRegressor
import pandas as pd

def model():
    # Load the data
    X = MLDataNum[Predictors].values
    y = MLDataNum[Target_Variable].values

    # Create the XGBRegressor model
    model = XGBRegressor(max_depth=2,
    learning_rate=0.1,
    n_estimators=1000,
    objective='reg:linear',
    booster='gbtree')

    # Train the model
    model.fit(X, y)
    # Predict the price based on the given predictors
    predictors = np.array([
        float(txtFirstNum.get()),  # carat
        0,    # cut_Fair
        0,    # cut_Good
        1,    # cut_Ideal
        0,    # cut_Premium
        0,    # cut_Very Good
        0,    # color_D
        1,    # color_E
        0,    # color_F
        0,    # color_G
        0,    # color_H
        0,    # color_I
        0,    # color_J
        0,    # clarity_I1
        0,    # clarity_IF
        0,    # clarity_SI1
        1,    # clarity_SI2
        0,    # clarity_VS1
        0,    # clarity_VS2
        0,    # clarity_VVS1
        0     # clarity_VVS2
    ]).reshape(-1, 21)

    predicted_price = model.predict(predictors)[0]
    txtResult.set(f'Predicted Price: ${predicted_price}.')


# Create the GUI
window = Tk()
window.title("Calculator")

lblCarat = Label(window, text="Carat: ")
lblCarat.grid(row=0, column=0, pady=(20, 10))

lblClarity = Label(window, text="Clarity: ")
lblClarity.grid(row=1, column=0, pady=(20, 10))


txtFirstNum = StringVar()
entFirstNum = Entry(window, textvariable=txtFirstNum, width=5)
entFirstNum.grid(row=0, column=1, padx=10, pady=(20, 10))

txt2Num = StringVar()
ent2Num = Entry(window, textvariable=txt2Num, width=5)
ent2Num.grid(row=1, column=1, padx=10, pady=(20, 10))


btnPlus = Button(window, text="Calculate", command=model)
btnPlus.grid(row=0, column=2, pady=(10, 0))

txtResult = StringVar()
entResult = Entry(window, state="readonly", textvariable=txtResult, width=25)
entResult.grid(row=4, column=2, columnspan=2, pady=(10, 10))

window.mainloop()
