from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import json

file_path = 'out1.json'
count=0
X=[]
y=[]
count1=0
count2=0
count3=0
import random
with open(file_path, 'r', encoding='utf-8') as file:
    for line_number, line in enumerate(file, 1):
        try:
            json_obj = json.loads(line.strip())
            x_=[]
            if 1 <= json_obj['output']['time'] < 2:
                count1 += 1
            if 2 <= json_obj['output']['time'] < 3:
                count2 += 1
            if 3 <= json_obj['output']['time']:
                count3 += 1
            if  json_obj['output']['time']!=1:
                for keys in json_obj['profile'].keys():

                    # if keys=='PCIe Write Requests to BAR1 [Requests]' or keys=='SM Issue [Throughput %]' or keys=='SM Active [Throughput %]' or keys=='SYS Clock Frequency [MHz]' or keys=='GPC Clock Frequency [MHz]' or keys=='GR Active [Throughput %]' \
                    #     or keys=='Compute Warps in Flight [Throughput %]' or keys=='Compute Warps in Flight [Avg]' or keys=='Compute Warps in Flight [Avg Warps per Cycle]' or keys=='Unallocated Warps in Active SMs [Throughput %]' or keys=='Unallocated Warps in Active SMs [Avg]' or keys=='Unallocated Warps in Active SMs [Avg Warps per Cycle]' or keys=='DRAM Read Bandwidth [Throughput %]' or keys=='DRAM Write Bandwidth [Throughput %]':
                        x_.append(json_obj['profile'][keys])


                for keys in json_obj['output'].keys():
                    if keys=='time':
                        y.append(json_obj['output']['time'])
                    else:
                        x_.append(json_obj['output'][keys])
                X.append(x_)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON at line {line_number}: {e}")
print(count1,count2,count3)
X=np.array(X)
y=np.array(y)
print(type(X),type(y),X.shape,y.shape)
# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 初始化随机森林回归器
n_estimators_range=[int(x) for x in np.linspace(start=50,stop=3000,num=60)]
max_depth_range=[int(x) for x in np.linspace(10,500,num=50)]
max_depth_range.append(None)
min_samples_split_range=[2,5,10]
min_samples_leaf_range=[1,2,4,8]
mse=100000000
record=[]



rf_regressor = RandomForestRegressor(n_estimators=500, random_state=42)
# 训练随机森林回归器
rf_regressor.fit(X_train, y_train)
import joblib
joblib.dump(rf_regressor, 'rf.pkl')
rf_regressor_=joblib.load('rf.pkl')
# # 在测试集上进行预测
# count=0

y_pred = rf_regressor.predict(X_test)
print(y_test,y_pred)
# 计算并打印均方误差
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 如果你想查看特征重要性
feature_importances = rf_regressor.feature_importances_
print(f"Feature Importances: {feature_importances}")

