# ARIMA-TimeSeries-Forecast

공공데이터 포털에 있는 2022년까지의 자료를 바탕으로 ~ 2024년까지 향후 2년에 대한 수요를 예측합니다

ARIMA 및 계절별 ARIMA 기법 탐색 후 시계열 데이터 사용


## 데이터 세트 설명

### 한국가스공사_판매현황

연도별 도시가스(주택/업무난방, 일반용, 산업용, 냉난방공조용, 수송용, 기타, 도시가스합계) 및 발전용 판매현황에 대한 데이터로, 단위는 천톤입니다.

데이터 세트 출처([https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data](https://www.data.go.kr/data/3074325/fileData.do))

데이터 목록
* gas.csv


<!-- 목차 -->

# 차 례

| 번호 | 내용                                             |
|------|--------------------------------------------------|
| 1  | [데이터 준비](#1)                                   |
| 2  | [시계열 데이터 시각화](#2)                          |
| 3  | [시계열 데이터를 고정적으로 만들기](#3)              |
| 4  | [상관관계 및 자기상관 차트 그리기](#4)               |
| 5  | [ARIMA 모델 또는 계절 ARIMA 구축](#5)               |
| 6  | [모델을 사용하여 예측하기](#6)                      |





<!-- intro -->
<div id="1">

# 1.데이터 준비

## 1.1 임포트

```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
```
## 1.2 로드

```python
df = pd.read_csv('gas.csv',encoding = 'cp949')

# 각종 데이터 확인
df
df.columns
df.describe()
df.info()
```
## 1.3 데이터 처리

```python

연월	도시가스용_주택과 업무난방용	도시가스용_일반용	도시가스용_산업용	도시가스용_냉난방공조용	도시가스용_수송용	도시가스용_기타	도시가스용_합계	발전용
0	2017-01	1498	189	617	74	81	116	2575	1478
1	2017-02	1346	162	549	61	67	91	2275	1313
2	2017-03	1051	156	564	39	76	74	1960	1260
3	2017-04	605	114	469	12	65	44	1309	823
4	2017-05	333	104	428	16	69	45	994	741
...	...	...	...	...	...	...	...	...	...
58	2021-11	850	157	640	31	74	102	1854	1515
59	2021-12	1485	195	717	70	79	165	2711	1563
60	2022-01	1804	187	678	77	67	172	2985	1684
61	2022-02	1585	163	577	63	56	155	2600	1578
62	2022-03	1137	143	582	35	58	100	2055	1811
63 rows × 9 columns
```

데이터 확인 결과 8가지 열과 63 rows × 9 columns 이루어진 데이터이다.

* 연월의 경우 object이므로  datetime으로 변환
* 인덱스를 '연월'로 변경

```python
df['연월'] = pd.to_datetime(df['연월'])

0   연월               63 non-null     datetime64[ns]

df.set_index('연월', inplace = True)
```


</div>

<div id="2">

# 2.시계열 데이터 시각화

## 2.1 데이터 시각화

```python
plt.rcParams['font.family'] = 'Malgun Gothic'
df.plot()
```

OutPut

![image](https://github.com/plintAn/ARIMA-TimeSeries-Forecast/assets/124107186/1c4ee32a-8833-461b-b6d8-1d554ac5f847)

간단하게 '연월'에 대한 시각화를 진행하였고,


</div>

<div id="3">

# 3.시계열 데이터를 고정적으로 만들기

## 3.1 시계열 데이터 안정성(정상성) 테스트 ADF(Augmented Dickey-Fuller)

* 시계열 데이터 안정성: 시간이 지나도 평균, 분산, 공분산이 일정한 성질로 이런 안정성이 없으면 예측 모델의 성능이 저하될 수 있다
* 대부분의 시계열 분석: 데이터의 안정성을 전제로 진행.
* ADF 테스트: 시계열 데이터의 안정성을 판별하는 통계적 방법. p-값이 0.05보다 작을 경우 데이터는 안정적이라 판단.

```python
# 귀무가설(Ho): 데이터는 비정상성을 갖는다.
# 대안가설(H1): 데이터는 정상성을 갖는다.

def adfuller_test(column_data):
    result = adfuller(column_data)
    labels = ['ADF 테스트 통계', 'p-값', '#사용된 지연', '사용된 관찰 수']
    
    print("\n" + "-"*50)
    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))
        
    if result[1] <= 0.05:
        print("귀무가설을 기각할 만큼의 충분한 증거가 있다. 데이터는 단위근이 없어 안정적이다.")
    else:
        print("귀무 가설에 대한 약한 증거, 시계열에는 단위근이 있어 고정적이지 않음을 나타낸다.")

```

다음과 같이 호출하여 결과를 확인할 수 있다.

OutPut

```python
adfuller_test(df['도시가스용_주택과 업무난방용'])

--------------------------------------------------
ADF 테스트 통계 : -1.0471364875683
p-값 : 0.7356867001782581
#사용된 지연 : 11
사용된 관찰 수 : 51
귀무 가설에 대한 약한 증거, 시계열에는 단위근이 있어 고정적이지 않음을 나타낸다.
```

실행 결과 '발전용' 열을 제외하고 모두 0.05보다 높은 값으로 차분이 필요해 보인다.


## 3.2 정상성 만족을 위한 차분

정상성 만족을 위해 열의 데이터에 대해 이전 행과의 차이(차분)를 계산하고 그 결과를 새로운 열에 저장하는 작업합니다.


```python
# 각 컬럼에 대해서
for column in df.columns:
    # 현재 행의 값에서 이전 행의 값을 뺀 차분 값을 새로운 컬럼에 저장
    df[f"{column} 차분"] = df[column] - df[column].shift(1)
```

OutPut

```python
	도시가스용_주택과 업무난방용	도시가스용_일반용	도시가스용_산업용	도시가스용_냉난방공조용	도시가스용_수송용	도시가스용_기타	도시가스용_합계	발전용	도시가스용_주택과 업무난방용 차분	도시가스용_일반용 차분	도시가스용_산업용 차분	도시가스용_냉난방공조용 차분	도시가스용_수송용 차분	도시가스용_기타 차분	도시가스용_합계 차분	발전용 차분
연월																
2017-01-01	1498	189	617	74	81	116	2575	1478	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN
2017-02-01	1346	162	549	61	67	91	2275	1313	-152.0	-27.0	-68.0	-13.0	-14.0	-25.0	-300.0	-165.0
2017-03-01	1051	156	564	39	76	74	1960	1260	-295.0	-6.0	15.0	-22.0	9.0	-17.0	-315.0	-53.0
2017-04-01	605	114	469	12	65	44	1309	823	-446.0	-42.0	-95.0	-27.0	-11.0	-30.0	-651.0	-437.0
2017-05-01	333	104	428	16	69	45	994	741	-272.0	-10.0	-41.0	4.0	4.0	1.0	-315.0	-82.0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
2021-11-01	850	157	640	31	74	102	1854	1515	465.0	31.0	68.0	7.0	-2.0	17.0	586.0	128.0
2021-12-01	1485	195	717	70	79	165	2711	1563	635.0	38.0	77.0	39.0	5.0	63.0	857.0	48.0
2022-01-01	1804	187	678	77	67	172	2985	1684	319.0	-8.0	-39.0	7.0	-12.0	7.0	274.0	121.0
2022-02-01	1585	163	577	63	56	155	2600	1578	-219.0	-24.0	-101.0	-14.0	-11.0	-17.0	-385.0	-106.0
2022-03-01	1137	143	582	35	58	100	2055	1811	-448.0	-20.0	5.0	-28.0	2.0	-55.0	-545.0	233.0
63 rows × 16 columns
```

* 차분을 새로운 열에 추가

```python
# 데이터프레임의 모든 열에 대해 반복합니다.
for column in df.columns:
    # 현재 열(column)의 값에서 12행 전의 값을 빼서 계절 차분을 계산합니다.
    # 그 결과를 "[현재 열 이름] 계절 차분"이라는 이름의 새로운 열에 저장합니다.
    df[f"{column} 계절 차분"] = df[column] - df[column].shift(12)
```

사실 '발전용'은 차분이 필요없는 상태지만, 번거로우므로 진행합니다. 다만 추후 order=(p,d,q)에서 d=0값으로 조절하면 됩니다.

* 계절 차분은 시계열 데이터에서 일정한 시간 간격(예: 매년, 매분기)으로 반복되는 계절성 패턴을 제거하는 방법이다.
* 그래프에서도 확인 할 수 있겠지만 날씨가 추운 계절일 때 수요가 급증하는 것을 확인할 수 있다.

예를 들어, 월간 데이터에서 매년 동일한 달 사이의 값을 뺄 때 사용된다.

```python
def adfuller_test_pvalue(data, column_name=''):
    result = adfuller(data)
    p_value = result[1]
    print(f"\n{column_name} 계절 차분의 ADF 테스트 p-값: {p_value:.5f}")  # 소수점 아래 5자리까지 출력

for column in columns[:]:
    seasonal_diff = df[column] - df[column].shift(12)
    adfuller_test_pvalue(seasonal_diff.dropna(), column)
```

* 1차 ADF 결과 통과(5/8)
* 도시가스용_주택과 업무난방용 √
* 도시가스용_산업용 √
* 도시가스용_냉난방공조용 √
* 도시가스용_기타 √
* 도시가스용_합계 √
* 발전용 √

OutPut

```python
도시가스용_주택과 업무난방용 계절 차분의 ADF 테스트 p-값: 0.00029

도시가스용_일반용 계절 차분의 ADF 테스트 p-값: 0.33808

도시가스용_산업용 계절 차분의 ADF 테스트 p-값: 0.00088

도시가스용_냉난방공조용 계절 차분의 ADF 테스트 p-값: 0.00002

도시가스용_수송용 계절 차분의 ADF 테스트 p-값: 0.07749

도시가스용_기타 계절 차분의 ADF 테스트 p-값: 0.09955

도시가스용_합계 계절 차분의 ADF 테스트 p-값: 0.00009

발전용 계절 차분의 ADF 테스트 p-값: 0.01111
```

### 2.5 2차 계절 차분



```python
def adfuller_test_pvalue(data, column_name=''):
    result = adfuller(data)
    p_value = result[1]
    print(f"\n{column_name} 계절 차분의 ADF 테스트 p-값: {p_value:.5f}")  # 소수점 아래 5자리까지 출력

columns = ['도시가스용_일반용', '도시가스용_수송용', '도시가스용_기타']

for column in columns[:]:
    seasonal_diff = df[column] - df[column].shift(12)
    adfuller_test_pvalue(seasonal_diff.dropna(), column)

```

다음과 같이 2차 차분 진행 후에도 정상성을 만족하지 않는다. 사실 2차 차분 이상의 진행은 다음과 같은 이유로 잘 진행하지 않는다.

* 모델 복잡성 증가: 더 높은 차수의 차분을 취하면, 모델의 복잡성이 증가하게 되어 분석과 예측의 어려움이 생길 수 있다.
* 과도한 단순화: 2차 차분 이상을 취하면 데이터의 원래 패턴이나 구조를 과도하게 단순화시키게 되어 중요한 정보나 특성을 잃을 수 있다.

OutPut

```python
도시가스용_일반용 계절 차분의 ADF 테스트 p-값: 0.33808

도시가스용_수송용 계절 차분의 ADF 테스트 p-값: 0.07749

도시가스용_기타 계절 차분의 ADF 테스트 p-값: 0.09955
```



</div>

<div id="4">

# 4.상관관계 및 자기상관 차트 그리기

자기 상관 그래프를 그리는 이유는 크게 4가지로 다음과 같다.

* 패턴 파악: 시계열 데이터 내의 계절성, 트렌드 등의 패턴을 감지
* 모델 파라미터 결정: ARIMA와 같은 모델의 매개변수를 결정하는 데 도움
* 정상성 확인: 시계열 데이터의 정상성 여부를 평가
* 이상치 탐지: 예상치 못한 이상치를 감지

```python
# 필요한 라이브러리를 불러옵니다.
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt

# 자기상관 그래프를 그릴 컬럼들의 리스트입니다.
columns = ['도시가스용_주택과 업무난방용','도시가스용_산업용', '도시가스용_기타', '도시가스용_합계', '발전용']

# 각 컬럼에 대해 자기상관 그래프를 그립니다.
for column in columns:
    # 그래프 크기 설정
    plt.figure(figsize=(12,6))
    # 그래프의 제목 설정
    plt.title(f"{column}의 자기상관 그래프")
    # 자기상관 그래프 그리기
    autocorrelation_plot(df[column])
    # 그래프 출력
    plt.show()
```

왼쪽부터 '도시가스용_주택과 업무난방용','도시가스용_산업용', '도시가스용_기타', '도시가스용_합계', '발전용' 순이다.

OutPut

![image](https://github.com/plintAn/ARIMA-TimeSeries-Forecast/assets/124107186/90a1cb0a-16ed-4444-9886-56c481f7c95c)


</div>

<div id="5">

# 5.ARIMA 모델 또는 계절 ARIMA 구축

## 5.1 시계열 데이터의 계절성 차분 ACF 및 PACF 분석

* SARIMA는 Seasonal AutoRegressive Integrated Moving Average의 약자로, 계절성을 갖는 시계열 데이터를 분석하고 예측하기 위해 사용된다.
* ARIMA의 기본 구조를 기반으로 하되, 추가적으로 계절성 요인을 고려한다.
* 계절성 요소는 특정 주기(예: 매월, 매년 등)로 반복되는 패턴을 확보합니다.

### 5.1RACFPACF를 보고 AR의 차수 p를 결정, ACF를 보고 MA의 차수 q를 결정

### 처음으로 통계적으로 유이하지 않는 값이 나타나는 지점(시점) 모두 1
* '도시가스용_주택과 업무난방용','도시가스용_산업용', '도시가스용_기타', '도시가스용_합계'

### 처음부터 정상성을 만족하고 있는 열은 0

* '발전용'은 0

PACF를 보고 AR의 차수 p를 결정합니다.

* PACF 그래프에서 처음으로 통계적으로 유의하지 않은 지점의 시차를 찾는다.
* (즉, 그래프의 파란 영역 안에 있는 첫 번째 시차) 이 시차는 AR의 차수 p로 사용된다

ACF를 보고 MA의 차수 q를 결정합니다.

* ACF 그래프에서 처음으로 통계적으로 유의하지 않은 지점의 시차를 찾습니다. 
* (즉, 그래프의 파란 영역 안에 있는 첫 번째 시차) 이 시차는 MA의 차수 q로 사용된다.

그래프 확인 결과 모두 AR의 차수 p값, MA 차수 q값은 1이다.

OutPut

![image](https://github.com/plintAn/ARIMA-TimeSeries-Forecast/assets/124107186/d8591690-d3a3-41bd-9d47-0b3ad03cfc72)

![image](https://github.com/plintAn/ARIMA-TimeSeries-Forecast/assets/124107186/1047c2fa-5641-4b89-90ee-06abcd0678b1)

![image](https://github.com/plintAn/ARIMA-TimeSeries-Forecast/assets/124107186/9d3906e0-7fe7-4796-a6c4-3a426673cd1b)

![image](https://github.com/plintAn/ARIMA-TimeSeries-Forecast/assets/124107186/b377a444-b7b7-445b-80db-66d518b7b197)

![image](https://github.com/plintAn/ARIMA-TimeSeries-Forecast/assets/124107186/15fb5a06-4863-496a-af73-3c9afbd91e47)

## 5.2 ARIMA 계절 차분분 모델링 및 결과 저장

'발전용' 차분에 대해서는 제외하고 모델링을 진행한다.

```python
from statsmodels.tsa.arima.model import ARIMA

# 분석 대상인 열(column)들을 정의
columns = ['도시가스용_주택과 업무난방용','도시가스용_산업용', '도시가스용_기타', '도시가스용_합계', '발전용']

# 각 열(column)에 대한 ARIMA 모델 결과를 저장하기 위한 딕셔너리 생성
models = {}

# 각 열(column)에 대하여 ARIMA 모델 적용 및 훈련
for column in columns:
    if column == '발전용':
        order = (1,0,1)  # '발전용' 열에 대해 차분 값을 0으로 지정
    else:
        order = (1,1,1)  # 나머지 열에 대해 차분 값을 1로 지정
        
    # ARIMA 모델 설정
    model = ARIMA(df[column], order=order)
    
    # 모델 훈련
    model_fit = model.fit()
    
    # 훈련된 모델을 'models' 딕셔너리에 해당 열 이름을 key로 저장
    models[column] = model_fit
```

OutPut


```python
model_fit.summary()

SARIMAX Results
Dep. Variable:	발전용	No. Observations:	63
Model:	ARIMA(1, 0, 1)	Log Likelihood	-426.119
Date:	Thu, 31 Aug 2023	AIC	860.239
Time:	18:23:16	BIC	868.811
Sample:	01-01-2017	HQIC	863.610
- 03-01-2022		
Covariance Type:	opg		
coef	std err	z	P>|z|	[0.025	0.975]
const	1313.1464	69.940	18.775	0.000	1176.067	1450.226
ar.L1	0.4360	0.190	2.297	0.022	0.064	0.808
ma.L1	0.4745	0.179	2.654	0.008	0.124	0.825
sigma2	4.332e+04	8306.626	5.215	0.000	2.7e+04	5.96e+04
Ljung-Box (L1) (Q):	0.02	Jarque-Bera (JB):	0.18
Prob(Q):	0.88	Prob(JB):	0.91
Heteroskedasticity (H):	0.97	Skew:	-0.05
Prob(H) (two-sided):	0.95	Kurtosis:	2.75


```


</div>

<div id="6">



# 6.모델을 사용하여 예측하기

## 6.1 예측 시각화

```python
import statsmodels.api as sm
import matplotlib.pyplot as plt

columns = ['도시가스용_주택과 업무난방용','도시가스용_산업용', '도시가스용_기타', '도시가스용_합계', '발전용']

# 결과 및 예측값을 저장할 빈 딕셔너리
results_dict = {}
forecast_dict = {}

for column in columns:
    # 해당 열에 대해 SARIMAX 모델 적합
    model = sm.tsa.statespace.SARIMAX(df[column], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit(disp=-1)
    
    # 결과 저장
    results_dict[column] = results
    
    # 예측값 저장
    start = len(df) - 1  
    end = start + 13     
    forecast_dict[column] = results.predict(start=start, end=end, dynamic=True)

# 예측 그래프 출력
plt.figure(figsize=(12, 8))
for column in columns:
    plt.plot(df[column], label=column)
    plt.plot(forecast_dict[column], label=f'{column} Forecast', linestyle='--')

plt.legend()
plt.title("Forecasts for All Columns")
plt.show()
```

## 6.2 미래 데이터 생성 후 예측

```python
from pandas.tseries.offsets import DateOffset

# df의 마지막 날짜로부터 24개월(2년) 동안의 미래 날짜를 생성
# DateOffset을 사용하여 각 달마다의 날짜를 계산
future_dates = [df.index[-1] + DateOffset(months=x) for x in range(0, 24)]
```

```python
# 생성한 미래 날짜를 인덱스로 사용하고, 원래 데이터프레임(df)의 열(column) 구조를 유지하여 새로운 데이터프레임을 생성
future_datest_df = pd.DataFrame(index=future_dates[1:], columns=df.columns)
future_df=pd.concat([df,future_datest_df])

```

```python
# 미래 날짜들을 사용하여 새로운 데이터프레임 'future_df' 생성. 원래 데이터프레임(df)의 열(column) 구조를 유지함
future_df = pd.DataFrame(index=future_dates[1:], columns=df.columns)

# 원래의 데이터프레임 'df'와 새로운 'future_df'를 연결하여 확장된 데이터프레임 'df'로 업데이트
df = pd.concat([df, future_df])

```

```python
columns_to_convert = ['도시가스용_주택과 업무난방용', '도시가스용_산업용', '도시가스용_기타', '발전용']

for column in columns_to_convert:
    df[column] = df[column].astype(float)
```

## 6.3 미래 데이터 생성 후 예측 시각화

시계열 데이터를 예측하기 위해 SARIMAX 모델을 사용하며, 각 column에 대한 예측 값을 그래프로 나타낸다.

```python
plt.figure(figsize=(12, 8))

for column in columns_to_convert:
    model = sm.tsa.statespace.SARIMAX(df[column], order=(1, 1, 1), seasonal_order=(1,1,1,12))
    results = model.fit()
    
    df[f'{column}_forecast'] = results.predict(start=len(df)-24, end=len(df)-1, dynamic=True)
    plt.plot(df[column], label=column)
    plt.plot(df[f'{column}_forecast'], label=f'{column} Forecast', linestyle='--')

plt.legend()
plt.title("Forecasts for Each Column")
plt.show()

```

Loading ...

![final](https://github.com/plintAn/ARIMA-TimeSeries-Forecast/assets/124107186/9c65b99b-4d6b-43ea-b278-6c665d77646f)


OutPut

![image](https://github.com/plintAn/ARIMA-TimeSeries-Forecast/assets/124107186/86668019-fa7a-40a2-b178-ce334afe8d31)



</div>

<div id="7">

</div>

<div id="8">

