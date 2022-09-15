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


<pre><code></code></pre>
<pre><code></code></pre>
<pre><code></code></pre>
  
 



  

  
