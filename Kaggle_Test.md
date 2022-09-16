# -Click-Through Rate Prediction
- Data source：https://www.kaggle.com/competitions/avazu-ctr-prediction/overview
- Colab：https://colab.research.google.com/drive/1P9X9hZvHsFBf_yVv1MTZ95EONmRIoKTB?usp=sharing

##  - 資料讀取
訓練集資料有40,428,967筆，由於資料量過大，將資料切分多個區塊(每區塊100萬筆)，並從每個區塊隨機抽取20%資料。

<pre><code>chunksize = 10 ** 6
num_of_chunk = 0
train = pd.DataFrame()

for chunk in pd.read_csv(path_root+'train.gz', chunksize=chunksize):
num_of_chunk += 1
train = pd.concat([train, chunk.sample(frac=.20, replace=False, random_state=1)], axis=0)
print('Processing' + str(num_of_chunk))     

train.reset_index(inplace=True,drop=True)

train_len = len(train)</code></pre>

讀取test和submission的資料
<pre><code>test = pd.read_csv(path_root+"test.gz")
submission = pd.read_csv(path_root+"sampleSubmission.gz",index_col='id')
print('Train dataset:',train.shape)
print('Test dataset:',test.shape)
print('Submission:',submission.shape)</code></pre>

##  - 資料前處理
看各欄位資料型態
<pre><code>train.info()</code></pre>
![image](https://user-images.githubusercontent.com/46454532/190408847-20fabdb1-1e53-4d52-8bc4-6147e8c4d73a.png)

在討論區中，官方公布所有資料皆為類別型態，故將所有欄位轉為object，保留click欄位為int型態
<pre><code>train = train.astype(object)
train['click']=train['click'].astype('int')

test = test.astype(object)</code></pre>

hour欄位為時間，將hour轉為weekday(星期幾)與period(第幾小時)

將hour轉換為時段
<pre><code>def transfer_period(h):
    h = h[-2:]
    return int(h)

train['period'] = train.hour.apply(transfer_period)
test['period'] = test.hour.apply(transfer_period)</code></pre>

將hour轉換為星期幾
<pre><code>def transfer_day_of_week(h):
  date_time_obj = datetime.strptime(h[:-2],'%y%m%d')
  return date_time_obj.weekday()

train['weekday'] = train.hour.apply(transfer_day_of_week)
test['weekday'] = test.hour.apply(transfer_day_of_week)</code></pre>


刪除hour與id欄位
<pre><code>train.drop(['hour','id'],axis=1,inplace = True)
test.drop(['hour','id'],axis=1,inplace = True)</code></pre>

確認各欄位類別數量
<pre><code>train.describe(include='object').T</code></pre>
![image](https://user-images.githubusercontent.com/46454532/190469970-88ff1e8a-1283-45cd-a5e3-d26f14c74fd4.png)

許多欄位類別數量過高，將各欄位中的各類別再進行一次分類，依照各類別的平均點擊率來進行區分，分為12種類，平均點擊率為0的為第0類，平均點擊率為1的為第11類，剩餘的最高與最低切分為10等份，例如C1平均點擊率排除1與0後，最高值平均點擊率值為0.21125226，最低平均點擊率值為0.03066793，將會以此兩數值均等切分成10等份，成為第1~10類。
測試集的資料也會在此時轉換成新組別，若有類別是在訓練集未曾出現過的，將會保留為NaN。
<pre><code>def group_column_with_click_mean(column_name,input_train_dt,input_test_dt):
  print(column_name)
  group_mean_dt = input_train_dt.groupby(column_name,as_index=False)['click'].mean()
  group_mean_dt[column_name+"_group"]=0
  group_mean_dt[column_name+"_group"], cut_bin =pd.cut(group_mean_dt[(group_mean_dt["click"]<1) & (group_mean_dt["click"]>0)]["click"], bins =10, labels = list(range(1,11)), retbins = True)
  print(cut_bin)
  group_mean_dt[column_name+"_group"]=group_mean_dt[column_name+"_group"].astype('str')
  group_mean_dt.loc[group_mean_dt['click']==1,column_name+"_group"]=11
  group_mean_dt.loc[group_mean_dt['click']==0,column_name+"_group"]=0
  group_mean_dt[column_name+"_group"]=group_mean_dt[column_name+"_group"].astype('int')
  group_mean_dt.drop(['click'],axis=1,inplace = True)
  input_train_dt = input_train_dt.merge(group_mean_dt,on = column_name ,how = 'left')
  input_test_dt = input_test_dt.merge(group_mean_dt,on = column_name ,how = 'left')
  input_train_dt.drop([column_name],axis=1,inplace = True)
  input_test_dt.drop([column_name],axis=1,inplace = True)
  return input_train_dt, input_test_dt</code></pre>
 
列出需要轉成新組別的欄位
<pre><code>need_to_be_transfer_list=train.drop(["click"],axis=1).columns.to_list()
need_to_be_transfer_list</code></pre>
![image](https://user-images.githubusercontent.com/46454532/190471756-18cf015d-2c49-4cfa-8bb9-0e5750628cd3.png)

轉換各欄位成為新類別
<pre><code>for column in need_to_be_transfer_list:
  train, test = group_column_with_click_mean(column,train,test)</code></pre>
![image](https://user-images.githubusercontent.com/46454532/190526573-e496abf7-4429-4a52-9f75-5e29b6a071a9.png)
![image](https://user-images.githubusercontent.com/46454532/190526633-76c93b16-b2f8-49ec-ae5e-01a055e9a07b.png)

轉換後的訓練集
![image](https://user-images.githubusercontent.com/46454532/190526822-4214bc73-7c8d-4271-950f-12281df8759a.png)

轉換後的測試集
![image](https://user-images.githubusercontent.com/46454532/190526894-86acaaf7-1ebc-4889-8eca-ecfcf02546e1.png)

觀察測試集NaN數量，也就是冷啟動的影響程度
<pre><code>test.isnull().sum()</code></pre>
![image](https://user-images.githubusercontent.com/46454532/190527011-8daec01f-e0c9-444f-a1ba-5119a77c15c3.png)

將遺漏比例大於1%的欄位列出
<pre><code>test_null = test.isnull().sum()> len(test)*0.01
test_null_list= test_null[test_null].index.tolist()
test_null_list</code></pre>

刪除這些欄位
<pre><code>train.drop(test_null_list,axis=1,inplace = True)
test.drop(test_null_list,axis=1,inplace = True)</code></pre>

看刪除後的NaN數量
<pre><code>test.isnull().sum()</code></pre>
![image](https://user-images.githubusercontent.com/46454532/190527293-5a258e7c-ed0e-4455-a081-85436c49dc79.png)

將NaN填補該欄位頻次最高的類別
<pre><code>test = test.apply(lambda x:x.fillna(x.value_counts().index[0]))</code></pre>

確認填補狀況
<pre><code>test.isnull().sum()</code></pre>
![image](https://user-images.githubusercontent.com/46454532/190527382-2744fed8-43f6-4206-bfe0-abdd1403f71a.png)

看train與test各欄位種類
<pre><code>for inx, val in enumerate(train.columns.tolist()):
  print(inx,val,":", pd.unique(train[val]))</code></pre>
![image](https://user-images.githubusercontent.com/46454532/190527484-f48e9f15-a554-45d5-bfe7-703f2b275cfa.png)

<pre><code>for inx, val in enumerate(test.columns.tolist()):
  print(inx,val,":", pd.unique(test[val]))</code></pre>
![image](https://user-images.githubusercontent.com/46454532/190527510-ed9ed10b-626e-4f52-ae91-f370f49f2b18.png)

將所有資料型態轉換為int，因新分類為有序類別。
<pre><code>train = train.astype(int)
test = test.astype(int)</code></pre>

拆分feature與target matrix
<pre><code># This is feature matrix
X = train.loc[:,train.columns != "click"]

# This is target vector
y = train["click"]</code></pre>

切分train與validation set，依照train中click的比例切分
<pre><code>print("y:",Counter(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,stratify=y ,random_state=1)
print("y_train:",Counter(y_train),"y_test:",Counter(y_test))</code></pre>
![image](https://user-images.githubusercontent.com/46454532/190527841-c2a43f09-e245-4862-8796-4526a5fe37cf.png)

##  - 模型建立
建立score function以利評估模型
<pre><code>def score(m, x_train, y_train, x_test, y_test, train=True):
    if train:
        pred=m.predict(x_train)
        print('Train Result:\n')
        print(f"Accuracy Score: {accuracy_score(y_train, pred)*100:.2f}%")
        print(f"Precision Score: {precision_score(y_train, pred)*100:.2f}%")
        print(f"Recall Score: {recall_score(y_train, pred)*100:.2f}%")
        print(f"F1 score: {f1_score(y_train, pred)*100:.2f}%")
        print(f"Log loss: {log_loss(y_test,m.predict_proba(x_train))}")
        print(f"Confusion Matrix:\n {confusion_matrix(y_train, pred)}")
        print(plot_confusion_matrix(m,x_test,y_test,cmap=plt.cm.Blues))
        print('\n')
        print(classification_report(y_test,pred))
        

    elif train == False:
        pred=m.predict(x_test)
        print('Test Result:\n')
        print(f"Accuracy Score: {accuracy_score(y_test, pred)*100:.2f}%")
        print(f"Precision Score: {precision_score(y_test, pred)*100:.2f}%")
        print(f"Recall Score: {recall_score(y_test, pred)*100:.2f}%")
        print(f"F1 score: {f1_score(y_test, pred)*100:.2f}%")
        print(f"Log loss: {log_loss(y_test,m.predict_proba(x_test))}")
        print(f"Confusion Matrix:\n {confusion_matrix(y_test, pred)}")
        print(plot_confusion_matrix(m,x_test,y_test,cmap=plt.cm.Blues))
        print('\n')
        print(classification_report(y_test,pred))</code></pre>
        
使用RandomizedSearchCV來選擇最佳超參數，從以下數組中，隨機選擇成為組合進行訓練
<pre><code>n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=9)]
max_depth = [int(x) for x in np.linspace(1,15, num=15)]

learning_rate=[round(float(x),2) for x in np.linspace(start=0.01, stop=0.2, num=10)]
colsample_bytree =[round(float(x),2) for x in np.linspace(start=0.1, stop=1, num=10)]

random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'learning_rate': learning_rate,
               'colsample_bytree': colsample_bytree}
random_grid</code></pre>
![image](https://user-images.githubusercontent.com/46454532/190528366-5c124adb-d0bb-48b6-9995-1b6926b5304c.png)

使用3 fold的交叉驗證進行100次訓練，並使用gpu加速
<pre><code>xg_pre = XGBClassifier(random_state=42,tree_method = 'gpu_hist')

xg_random = RandomizedSearchCV( estimator = xg_pre, param_distributions=random_grid,
                              n_iter=100,scoring = 'neg_log_loss' ,cv=3, verbose=2, random_state=42, n_jobs=-1)
xg_random.fit(X_train,y_train)
xg_random.best_params_</code></pre>
![image](https://user-images.githubusercontent.com/46454532/190528463-88c5f698-26bb-420e-bdd6-08aea2b9f634.png)

使用最佳參數訓練模型
<pre><code>xg = XGBClassifier(tree_method = 'gpu_hist',colsample_bytree= 0.6, learning_rate=0.01, max_depth= 14, n_estimators=800)
xg=xg.fit(X_train,y_train)
score(xg, X_train, y_train, X_test, y_test, train=False)</code></pre>
![image](https://user-images.githubusercontent.com/46454532/190528952-863781a4-1aa6-4ee5-b050-44be26f55ef9.png)

重要特徵
<pre><code>from xgboost import plot_importance
plot_importance(xg)
plt.figure(figsize=(20, 15))
plt.show()</code></pre>
![image](https://user-images.githubusercontent.com/46454532/190529152-b65730cc-fbc2-4702-8c2d-9fc3650e5bab.png)

預測test匯出submission.csv
<pre><code>submission = pd.read_csv(path_root+"sampleSubmission.gz",index_col='id')
submission[submission.columns[0]] = xg.predict_proba(test)[:,1]
submission.to_csv(path_root+'submission.csv')</code></pre>

於Kaggle提交分數為 0.40747
![image](https://user-images.githubusercontent.com/46454532/190529364-5097ce80-303d-4fcc-ad5f-bebab14a3657.png)
