# DACON-Competition3

![image](https://user-images.githubusercontent.com/75110162/104088219-85efca00-52a8-11eb-84b4-071bf839f35f.png)

세번째로 참가한 DACON Competition, 여러 Pre-Trained Model들을 다뤄볼 수 있었던 기회 

![image](https://user-images.githubusercontent.com/75110162/104088265-cea78300-52a8-11eb-953e-bb6a1cd1dd3c.png)

- 뉴스의 Title과 Cotent가 주어지는데 해당 Content가 Title과 관련 있는 기사인지 아닌지 분류하는 __Binary Classification__ 과제 

- **방향**:  4가지의 PreTrained Model과 XGBoost를 Ensemble 하여 성능을 높이자! 

--------

### 모델1. PreTrained Model (KoBERT, KoGPT, KoELECTRA, KoBART) - KoBERT 모델 예시 

#### STEP1: 전처리: Title과 Content를 합침
단순히 내용만으로 분류하는 것보다 해당 기사의 제목과 내용을 합치는 것이 정확도가 높게 나왔음
``` python
train['title_content']=train['title']+' '+train['content']
test['title_content']=test['title']+' '+test['content']
```
#### STEP2: BERTClassifier Class 
``` python
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(pooler)
```

#### STEP3: Dataset
``` python
class BERTDataset1(Dataset):
    def __init__(self, dataset, sent_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i])

    def __len__(self):
        return (len(self.sentences))

test_x = test.title_content.values

dataset_test = []
for i in range(test_x.shape[0]):
    dataset_test.append([test_x[i]])

data_test = BERTDataset1(dataset_test, 0, tok, 128, True, False)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=64, num_workers=5)
```

#### STEP4: Train 데이터로 학습시킨 모델 load후 Test 데이터 Predict
``` python
bert=torch.load('/content/drive/MyDrive/open/Bert_model.pt').to(device)

bert.eval()
for batch_id, (token_ids, valid_length, segment_ids) in enumerate(tqdm(test_dataloader)):
    with torch.no_grad():
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        out = bert(token_ids, valid_length, segment_ids)
        if(batch_id==0):
            preds=out
        else:
            preds = torch.vstack((preds,out))
bert_preds = preds
```

---------------

### 모델2. XGBoost
Concept:  Logisitic Regression, LSTM, CNN, FastText 모델을 Stacking Ensemble 하여 최종 모델로 XGBoost 

#### STEP1: Logisitic Regression
- Mecab Tokenizer + CounterVectorizer + Logistic Regression + KFold

``` python
count_char_train = count_char_vec.transform(train.title_content.values)
count_word_train = count_word_vec.transform(train.title_content.values)
count_char_test = count_char_vec.transform(test.title_content.values)
count_word_test = count_word_vec.transform(test.title_content.values)

x_train1 = sparse.hstack([count_char_train, count_word_train])
x_test1 = sparse.hstack([count_char_test, count_word_test])

def runLR(train_X,train_y,test_X,test_y,test_X2):
    model=LogisticRegression()
    model.fit(train_X,train_y)
    pred_test_y=model.predict_proba(test_X)
    pred_test_y2=model.predict_proba(test_X2)
    return pred_test_y, pred_test_y2, model
    
cv_scores=[]
train_y = np.array(train['info'])
pred_full_test = 0
pred_train=np.zeros([train.shape[0],2])

kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)

for dev_index, val_index in kf.split(train,train_y):
    dev_X, val_X = x_train1.tocsc()[dev_index], x_train1.tocsc()[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_val_y, pred_test_y, model = runLR(dev_X, dev_y, val_X, val_y, x_test1.tocsc())
    pred_train[val_index,:] = pred_val_y
    pred_full_test = pred_full_test + pred_test_y
    cv_scores.append(metrics.log_loss(val_y, pred_val_y))
print("Mean cv score : ", np.mean(cv_scores))
pred_full_test = pred_full_test / 5.

x_train["lr_count_0"] = pred_train[:,0]
x_train["lf_count_1"] = pred_train[:,1]
x_test["lr_count_0"] = pred_full_test[:,0]
x_test["lf_count_1"] = pred_full_test[:,1]
```

#### STEP2: LSTM, CNN 
- Keras Tokenizer fitting, Train 데이터로 학습시킨 CNN,LSTM 모델로 Predict 

``` python
from keras.preprocessing import text, sequence

data_x = train.title_content.values
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(data_x)
tokenized_train = tokenizer.texts_to_sequences(data_x)
train_x = sequence.pad_sequences(tokenized_train, maxlen=128)

data_test_x = test.title_content.values
tokenized_test = tokenizer.texts_to_sequences(data_test_x)
test_x = sequence.pad_sequences(tokenized_test, maxlen=128)

from keras.models import load_model

lstm_model = load_model("/content/drive/MyDrive/open/CuDNNLSTM.h5")
cnn_model = load_model("/content/drive/MyDrive/open/CNN.h5")

x_train['LSTM'] = lstm_model.predict(train_x)
x_test['LSTM'] = lstm_model.predict(test_x)
x_train['CNN'] = cnn_model.predict(train_x)
x_test['CNN'] = cnn_model.predict(test_x)
```

### STEP3: FastText

FACEBOOK 의 FastText에서 제공하는 Unsupervised Learning 을 통하여 train data set을 학습시킴
이후, 학습된 FastText Model로 각 문장들을 임베딩 하였고 이를 Feature로 활용

``` python
train['text'].to_csv('sample_file.txt',index=False, header=None, sep="\t")
model_ft = fasttext.train_unsupervised('sample_file.txt', minCount=2, minn=2, maxn=10,dim=300)

def sent2vec(s):
    words = nltk.tokenize.word_tokenize(s)
    #words = [k.stem(w) for w in words]
    #words = [w for w in words if not w in string.digits]
    #words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model_ft[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v

xtrain_ft = np.array([sent2vec(x) for x in train['text']])
xtest_ft = np.array([sent2vec(x) for x in test['text']])

train_ft=pd.DataFrame(xtrain_ft)
train_ft.columns = ['ft_vector_'+str(i) for i in range(xtrain_ft.shape[1])]

test_ft=pd.DataFrame(xtest_ft)
test_ft.columns = ['ft_vector_'+str(i) for i in range(xtrain_ft.shape[1])]

train = pd.concat([train, train_ft], axis=1)
test = pd.concat([test, test_ft], axis=1)
```
#### STEP3: XGBoost 
총 **_430개의 Feature_** 를 Extract 하였고 XGBoost 모델로 Classification 학습했다. 
``` python
cols_to_drop = ['index', 'text']
train_X = train.drop(cols_to_drop+['author'], axis=1)
train_y=train['author']
test_index = test['index'].values
test_X = test.drop(cols_to_drop, axis=1)
xgb_preds=[]
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2020)

for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    
    dtrain = xgb.DMatrix(dev_X,label=dev_y)
    dvalid = xgb.DMatrix(val_X, label=val_y)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 3
    param['silent'] = 1
    param['num_class'] = 5
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.3
    param['seed'] = 0
    param['tree_method'] = 'gpu_hist'

    model = xgb.train(param, dtrain, 2000, watchlist, early_stopping_rounds=50, verbose_eval=20)

    xgtest2 = xgb.DMatrix(test_X)
    xgb_pred = model.predict(xgtest2, ntree_limit = model.best_ntree_limit)
    xgb_preds.append(list(xgb_pred))

```

#### STEP4: 결과
![image](https://user-images.githubusercontent.com/75110162/103353149-622fc580-4aeb-11eb-8589-91f6cf7f35eb.png)

0.14877 의 loss로 3등으로 대회를 마쳤다. 이는 초기 모델인 LSTM보다 훨씬 좋은 SCORE였다. Kaggle에서 왜 XGBoost가 인기 있는 모델인지 다시 한번 알 수 있었다.

그리고 운이 좋게도 1등,2등 을 하신 분들이 대회 규칙으로 인하여 수상에서 제외되고 3등인 내가 최종 1등이 되었다. 

![image](https://user-images.githubusercontent.com/75110162/103353628-a66f9580-4aec-11eb-90c6-d206296f9b89.png)


#### SELF 피드백 

- 전처리를 안하는 쪽이 loss를 줄여나가는 데에 분명 도움이 되었다. 그러나 아예 안하는 것은 최선은 아니었을 것 같다. 너무 모델링에만 집중한 것은 아니었을까
- XGBoost 의 Feature로 Neural Network Model을 사용하지 않았다. 다른 수상자 분들의 코드는 CNN이나 LSTM을 Stacking 했을 때 Feature Importance가 높았다. 

