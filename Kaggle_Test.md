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

