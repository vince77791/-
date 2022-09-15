# -Click-Through Rate Prediction
- Data source：https://www.kaggle.com/competitions/avazu-ctr-prediction/overview

##  - 資料讀取
train資料有40,428,967筆，由於資料量過大，將資料切分多個區塊，並從每個區塊隨機抽取20%資料。

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

hour欄位為時間，本資料集train為10天資料，test為第11天資料，基本上hour欄應無意義，將hour轉為weekday(星期幾)與period(第幾小時)

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

許多欄位類別數量過高，將各欄位中的各類別再進行一次分類，依照各類別的平均點擊率來進行區分，分為12種類，平均點擊率為0的為第0類，平均點擊率為1的為第11類，其餘將剩餘的最高與最低切分為10等份，例如C1平均點擊率排除1與0後，最高為0.5，最低為0.1，將會以此兩數值均等切分成10等份，成為第1~10類。
test set的資料也會在此時轉換成新組別，若有類別是在train set未曾出現過的，將會保留為NaN。
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

<pre><code></code></pre>
<pre><code></code></pre>
<pre><code></code></pre>
<pre><code></code></pre>
  
 



  

  
