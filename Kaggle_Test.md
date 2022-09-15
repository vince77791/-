# -Click-Through Rate Prediction
  - source：https://www.kaggle.com/competitions/avazu-ctr-prediction/overview

##  - 資料讀取
  train資料有40,428,967筆，由於資料量過大，將資料切分多個區塊，並從每個區塊隨機抽取15%資料。

<pre><code>chunksize = 10 ** 6
num_of_chunk = 0
train = pd.DataFrame()
    
for chunk in pd.read_csv(path_root+'train.gz', chunksize=chunksize):
    num_of_chunk += 1
    train = pd.concat([train, chunk.sample(frac=.10, replace=False, random_state=1)], axis=0)
    print('Processing' + str(num_of_chunk))     
    
train.reset_index(inplace=True,drop=True)

train_len = len(train)</code></pre>
  
  讀取test和submission的資料
  <pre><code>test = pd.read_csv(path_root+"test.gz")
  submission = pd.read_csv(path_root+"sampleSubmission.gz",index_col='id')
  print('Train dataset:',train.shape)
  print('Test dataset:',test.shape)
  print('Submission:',submission.shape)</code></pre>
  
  看各欄位資料型態
  <pre><code>train.info()</code></pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4042897 entries, 0 to 4042896
Data columns (total 24 columns):
 #   Column            Dtype  
---  ------            -----  
 0   id                float64
 1   click             int64  
 2   hour              int64  
 3   C1                int64  
 4   banner_pos        int64  
 5   site_id           object 
 6   site_domain       object 
 7   site_category     object 
 8   app_id            object 
 9   app_domain        object 
 10  app_category      object 
 11  device_id         object 
 12  device_ip         object 
 13  device_model      object 
 14  device_type       int64  
 15  device_conn_type  int64  
 16  C14               int64  
 17  C15               int64  
 18  C16               int64  
 19  C17               int64  
 20  C18               int64  
 21  C19               int64  
 22  C20               int64  
 23  C21               int64  
dtypes: float64(1), int64(14), object(9)
memory usage: 740.3+ MB
  
