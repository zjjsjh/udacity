import numpy as np
import pandas as pd

#数据处理
def data_process(data, features):
	data.loc[data['Open'].isnull(), 'Open'] = 1
	data.loc[data['CompetitionDistance'].isnull(), 'CompetitionDistance'] = 0
	features.extend(['Store', 'CompetitionDistance', 'Promo', 'Promo2', 'SchoolHoliday'])
	
	#calculate log(sales) and log(customers)
	data.loc[data['isTrain'] == 1, 'SalesLog'] = np.log1p(data.loc[data['isTrain'] == 1]['Sales'])
	data.loc[data['isTrain'] == 1, 'CustomersLog'] = np.log1p(data.loc[data['isTrain'] == 1]['Customers'])

	#change StoreType / Assortment / StateHoliday values
	replace_dict = {'a': 1, 'b': 2, 'c':3, 'd':4, '0':0}
	data.replace(replace_dict, inplace=True)
	features.extend(['StoreType', 'Assortment', 'StateHoliday'])
	
	#Date column split
	data['Year'] = data['Date'].apply(lambda x: x.year)
	data['Month'] = data['Date'].apply(lambda x: x.month)
	data['Day'] = data.Date.dt.day  
	data['DayOfWeek'] = data.Date.dt.dayofweek 
	data['WeekOfYear'] = data['Date'].apply(lambda x: x.weekofyear)
	data['DayOfYear'] = data.Date.dt.dayofyear
	features.extend(['DayOfWeek', 'Month', 'Day', 'Year', 'WeekOfYear', 'DayOfYear'])
	
	#competition open time in months
	data["CompetitionOpenSinceYear"][(data["CompetitionDistance"] != 0) & (data["CompetitionOpenSinceYear"].isnull())] = 1900#data['CompetitionOpenSinceYear'].median()
	data["CompetitionOpenSinceMonth"][(data["CompetitionDistance"] != 0) & (data["CompetitionOpenSinceMonth"].isnull())] = 1#data['CompetitionOpenSinceMonth'].median()
	data['CompetitionOpen'] = 12 * (data['Year'] - data['CompetitionOpenSinceYear']) +\
							(data['Month'] - data['CompetitionOpenSinceMonth'])
	features.append('CompetitionOpen')
	
	#promo open time in months
	data['PromoOpen'] = 12 * (data['Year'] - data['Promo2SinceYear']) +\
						(data['WeekOfYear'] - data['Promo2SinceWeek']) / 4.0
	data.loc[data['Promo2SinceYear'] == 0, 'PromoOpen'] = 0
	features.append('PromoOpen')
	
	#promo month
	month2str = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', \
				 7: 'Jul', 8: 'Aug', 9: 'Sept', 10: 'Oct', 11: 'Nov', 12: 'Dec'}  
	data['monthStr'] = data['Month'].map(month2str)
	data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
	data['IsPromoMonth'] = 0
	for interval in data['PromoInterval'].unique():
		if interval == interval:  
			for month in interval.split(','):
				data.loc[(data.monthStr == month) & (data.PromoInterval == interval), 'IsPromoMonth'] = 1 
	features.append('IsPromoMonth')
	
	#calculate number of schoolholidays this week, last week and next week
	SchoolHolidays = data.groupby(['Store','Year','WeekOfYear'])['SchoolHoliday'].sum().reset_index(name='holidays_thisweek')
	SchoolHolidays['holidays_lastweek'] = 0
	SchoolHolidays['holidays_nextweek'] = 0

	for store in SchoolHolidays.Store.unique().tolist():
		storeLen = len(SchoolHolidays[SchoolHolidays['Store'] == store])
		SchoolHolidays.loc[1:storeLen-1, 'holidays_lastweek'] = SchoolHolidays.loc[0:storeLen-2, 'holidays_thisweek'].values
		SchoolHolidays.loc[0:storeLen-2, 'holidays_nextweek'] = SchoolHolidays.loc[1:storeLen-1, 'holidays_thisweek'].values

	data = pd.merge(data, SchoolHolidays, how='left', on=['Store', 'Year', 'WeekOfYear'])
	features.extend(['holidays_thisweek', 'holidays_lastweek', 'holidays_nextweek'])
	
	#calculate average sales per store, sales per customer, customers per store,
	salesAvgPerStore = data[data.Open == 1].groupby(['Store'])['Sales'].mean()
	customersAvgPerStore = data[data.Open == 1].groupby(['Store'])['Customers'].mean()
	salesAvgPerCustomer = salesAvgPerStore / customersAvgPerStore        
	data = pd.merge(data, salesAvgPerStore.reset_index(name='SalesAvgPerStore'), how='left', on='Store')
	data = pd.merge(data, customersAvgPerStore.reset_index(name='CustomersAvgPerStore'), how='left', on='Store')
	data = pd.merge(data, salesAvgPerCustomer.reset_index(name='SalesAvgPerCustomer'), how='left', on='Store')
	features.extend(['SalesAvgPerStore', 'CustomersAvgPerStore', 'SalesAvgPerCustomer'])
		
	#calculate average sales per stateholiday, sales per schoolholiday
	data['isStateHoliday'] = 0
	data.loc[data.StateHoliday != 0, 'isStateHoliday'] = 1
	salesPerStateHoliday = data[data.isStateHoliday == 1].groupby(['Store'])['Sales'].mean()
	salesPerSchoolHoliday = data[data.SchoolHoliday == 1].groupby(['Store'])['Sales'].mean()
	data = pd.merge(data, salesPerStateHoliday.reset_index(name='SalesPerStateHoliday'), how='left', on='Store')
	data = pd.merge(data, salesPerSchoolHoliday.reset_index(name='SalesPerSchoolHoliday'), how='left', on='Store')
	features.extend(['SalesPerStateHoliday', 'SalesPerSchoolHoliday'])
		
	#calculate average and median sales and customers per store per day of week
	salesAvgPerDow = data.groupby(['Store', 'DayOfWeek'])['Sales'].mean()
	salesMedPerDow = data.groupby(['Store', 'DayOfWeek'])['Sales'].median()
	customersAvgPerDow = data.groupby(['Store', 'DayOfWeek'])['Customers'].mean()
	customersMedPerDow = data.groupby(['Store', 'DayOfWeek'])['Customers'].median()
	data = pd.merge(data, salesAvgPerDow.reset_index(name='SalesAvgPerDow'), how='left', on=['Store', 'DayOfWeek'])
	data = pd.merge(data, salesMedPerDow.reset_index(name='SalesMedPerDow'), how='left', on=['Store', 'DayOfWeek'])
	data = pd.merge(data, customersAvgPerDow.reset_index(name='CustomersAvgPerDow'), how='left', on=['Store', 'DayOfWeek'])
	data = pd.merge(data, customersMedPerDow.reset_index(name='CustomersMedPerDow'), how='left', on=['Store', 'DayOfWeek'])
	features.extend(['SalesAvgPerDow', 'SalesMedPerDow', 'CustomersAvgPerDow', 'CustomersMedPerDow'])
	
	data.fillna(0, inplace=True)
	return data

#rmspe
def weight(y):
    w = np.zeros(y.shape, dtype=float)
    idx = y != 0
    w[idx] = 1.0 / (y[idx]**2)
    return w

def rmspe_xg(yhat, y):
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = weight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return 'rmspe', rmspe

def rmspe(yhat, y):
    w = weight(y)
    rmspe = np.sqrt(np.mean(w * (yhat - y)**2))
    return rmspe

# create feature map
def create_feature_map(features):
    output = open('xgb.map', 'w')
    for i, f in enumerate(features):
        output.write('{0}\t{1}\tq\n' .format(i, f))
    output.close()

df_store = pd.read_csv("store.csv")
df_train = pd.read_csv("train.csv", parse_dates=[2])
df_test = pd.read_csv("test.csv", parse_dates=[3])

df_test['isTrain'] = 0
df_train['isTrain'] = 1
df_train_test = pd.concat([df_train, df_test])
df_train_test = df_train_test.loc[~((df_train_test.isTrain == 1) & (df_train_test.Sales == 0))]
all_df = pd.merge(df_train_test, df_store, how='left', on='Store')
features = []
data = data_process(all_df, features)
train_data = data[data['isTrain'] == 1]
test_data = data[data['isTrain'] == 0]
print(features)
#split train and test set
#timeDelta = pd.Timedelta(weeks=6)
timeDelta = test_data.Date.max() - test_data.Date.min()
maxDate = train_data.Date.max()
minDate = maxDate - timeDelta
testIn = train_data['Date'].apply(lambda x: (x <= maxDate and x >= minDate))
trainIn = testIn.apply(lambda x: (not x))
X_train = train_data[trainIn]
X_test = train_data[testIn]
y_train = train_data['SalesLog'][trainIn]
y_test = train_data['SalesLog'][testIn]
#xgboost
import xgboost as xgb
#randomly choose features and train xgboost models
import random
import copy

features_base = ['SalesAvgPerStore', 'CustomersAvgPerStore', 'Month', 'DayOfWeek', 'SalesAvgPerCustomer', 'CustomersAvgPerDow', 'SalesMedPerDow', 'SalesAvgPerDow', 'CompetitionDistance', 'WeekOfYear', 'Store', 'PromoOpen', 'CompetitionOpen', 'DayOfYear', 'Day']
features_random = ['CustomersMedPerDow', 'SalesPerSchoolHoliday', 'Year', 'holidays_thisweek', 'Promo', 'SalesPerStateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 'IsPromoMonth', 'Promo2', 'StateHoliday', 'holidays_nextweek', 'holidays_lastweek']

models_dict = {}
nround = 3000
params = {
    'objective': 'reg:linear',
    'booster': 'gbtree',
    'eta': 0.03,
    'max_depth': 10,
    'subsample': 0.8,
    'colsample_bytree': 0.6,
    'silent': 1,
    'seed':1301
}
for i in range(6, 10):
    for k in range(1, 6):
        #choose features
        feature_model = copy.deepcopy(features_base)
        j = 0
        model_info = {}
        while j < i:
            idx = random.randint(0, 13)
            if features_random[idx] in feature_model:
                continue
            feature_model.append(features_random[idx])
            j += 1
        filename = 'output/features/feature_{}{:0=3d}.txt'.format(15+i, k)
        outfile = open(filename, 'w')
        outfile.write(str(feature_model))
        outfile.close()
    
        model_info['features'] = feature_model
        #train xgboost model
        print('No.{} model'.format((i-5)*k))
        dtrain = xgb.DMatrix(X_train[feature_model], y_train)
        dtest = xgb.DMatrix(X_test[feature_model], y_test)
        
        watch_list = [(dtrain, 'train'), (dtest, 'eval')]
        gbm = xgb.train(params, dtrain, nround, watch_list, 
                       early_stopping_rounds=100, verbose_eval=500,
                       feval=rmspe_xg)        
        
        y_pred = gbm.predict(xgb.DMatrix(X_test[feature_model]))
        train_error = rmspe(np.expm1(y_pred), X_test['Sales'].values)
        model_info['train_error'] = train_error
        #test data prediction
        dvaild = xgb.DMatrix(test_data[feature_model])
        y_vaild = gbm.predict(dvaild) 
        
        filename = '{}{:0=3d}'.format(15+i, k)
        result = pd.DataFrame({'ID': test_data['Id'], 'Sales':np.expm1(y_vaild)})
        result[['ID']] = result[['ID']].astype(int)
        result.to_csv('output/test_predict/{}.csv'.format('test_'+filename), index=False)

        models_dict[filename] = model_info

df_models = pd.DataFrame(models_dict).T
df_models.to_csv('output/models_info.csv')