import streamlit as st
import numpy as np
import pandas as pd

# CSSを使って背景色を変更
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffebcd;#ffa500
    }
      [data-testid="stSidebar"] {
        background-color:#ffebcd:#ff8c00

     }
    </style>
    """,
    unsafe_allow_html=True

    
)


# CSVファイルのパス
file_path = '売上実績表.csv'

# CSVファイルの読み込み
train = pd.read_csv(file_path)
#タイトル
st.title ('How many do you make?')
# 画像 
st.image('b1.jpg')
image='b1.jpg'

# データの表示
print(train.columns)
train.columns = train.columns.str.strip()
print(train['Weather'])
print(train.head())  # Print the first few rows to inspect column names and data
columns_of_interest = ['Event','Temperature','Weather']
columns_present = [col for col in columns_of_interest if col in train.columns]
train_X = train[columns_present]

## 欠測値補完
from sklearn.impute import SimpleImputer
mean_imputer = SimpleImputer(strategy='mean')
train[['Temperature']] = mean_imputer.fit_transform(train[['Temperature']])

# 予測モデル構築
from sklearn.linear_model import LinearRegression
logreg = LinearRegression()
train_X = train[['Event','Temperature','Weather']]
train_y = train[['食パン','フランスパン','タルティーヌ','メロンパン', 'あんぱん', 'スコーン']]
logreg.fit(train_X, train_y)



## ラジオボタン
st.subheader('商品名')
ABC = st.radio('Please select one', ('食パン','フランスパン','タルティーヌ','メロンパン', 'あんぱん', 'スコーン'))

### 天気入力（ラジオボタン）
WeatherValue = st.sidebar.radio('Select Weather:Sunny:1,Cloudy:2,Rainy:3',[1,2,3])

### 行事入力（ラジオボタン）
EventValue = st.sidebar.radio('Select Event:休み:0,なし:1,あり:2',[0,1,2])

### 気温入力（スライドバー）
minValue_Temperature = int(np.floor(train['Temperature'].min()))
maxValue_Temperature = int(np.ceil(train['Temperature'].max()))
startValue_Temperature =int((maxValue_Temperature+minValue_Temperature)/2)
TemperatureValue = st.sidebar.slider('Temperature', min_value=minValue_Temperature, max_value=maxValue_Temperature, step=1, value=startValue_Temperature)

value_df = pd.DataFrame([WeatherValue,EventValue,TemperatureValue],index=['Event','Temperature','Weather']).T

### 予測
pred_probs = logreg.predict(value_df)

### 結果出力
import streamlit as st

# フォントサイズを大きくするテキストを追加
st.markdown(
    """
    <h2>本日の製造数は↓↓</h2>
    """,
    unsafe_allow_html=True
)

st.markdown(f"<h2 style='font-size:42px;'>{np.round(pred_probs[0][0])}</h2>", unsafe_allow_html=True)


#st.code('<div>HTML形式</div>', language='html')
import streamlit as st
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #ffebcd; :#ff8c00:
    }

    </style>
    """,
    unsafe_allow_html=True
)

date = st.sidebar.date_input('今日の日付')