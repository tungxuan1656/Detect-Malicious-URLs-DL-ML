#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
from keras.models import Sequential
from sklearn.metrics import accuracy_score, f1_score
from keras.layers import Dense, Dropout, Activation, Reshape
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


def loadData(file):
    with open(file, 'r', encoding="utf8") as f:
        data = f.readlines()
    result = []
    for d in data:
        d = d.strip()
        if (len(d) > 0):
            result.append(d)
    return result
def print_result(y_pred, y_test, clf_name):
    ACC = accuracy_score(y_pred, y_test)
    F1 = f1_score(y_pred, y_test, average='macro')
    print("%s\t(accuracy, f1) = (%.5f, %.5f)"%(clf_name, ACC, F1))


# In[ ]:


bad_requests = loadData('anomalousRequest.txt')
good_requests = loadData('normalRequest.txt')


# In[ ]:


all_requests = bad_requests + good_requests
y_bad = [1] * len(bad_requests)
y_good = [0] * len(good_requests)


# In[5]:


print("Total requests : ", len(all_requests))
print("Bad requests: ", len(bad_requests))
print("Good requests: ", len(good_requests))


# In[ ]:


vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char", sublinear_tf=True, ngram_range=(3, 3))


# Tách lần lượt normal_data và malicious_data thành 3 tập train, validation, test đều nhau

# In[ ]:


normal_train, normal_test, y_normal_train, y_normal_test = train_test_split(good_requests, y_good, test_size = 0.4, random_state = 22)
malicious_train, malicious_test, y_malicious_train, y_malicious_test = train_test_split(bad_requests, y_bad, test_size = 0.4, random_state = 22)
normal_test, normal_val, y_normal_test, y_normal_val = train_test_split(normal_test, y_normal_test, test_size = 0.5, random_state = 11)
malicious_test, malicious_val, y_malicious_test, y_malicious_val = train_test_split(malicious_test, y_malicious_test, test_size = 0.5, random_state = 11)


# In[ ]:


train = normal_train + malicious_train
y_train = y_normal_train + y_malicious_train
val = normal_val + malicious_val
y_val = y_normal_val + y_malicious_val
test = normal_test + malicious_test
y_test = y_normal_test + y_malicious_test


# In[9]:


print("Requests for Train data: %d\t (normal, malicious) = (%d, %d)"%(len(train), len(normal_train), len(malicious_train)))
print("Requests for Validation data: %d\t (normal, malicious) = (%d, %d)"%(len(val), len(normal_val), len(malicious_val)))
print("Requests for Test data: %d\t (normal, malicious) = (%d, %d)"%(len(test), len(normal_test), len(malicious_test)))
print("Use Trigram (n=3). Split Train:Validation:Test = 6:2:2\n")


# fit dữ liệu train và transform 3 tập dữ liệu

# In[10]:


vectorizer.fit(train)


# In[ ]:


X_train = vectorizer.transform(train)
X_val = vectorizer.transform(val)
X_test = vectorizer.transform(test)


# In[12]:


print("Shape of X_train: ", X_train.shape)
print("Shape of X_val: ", X_val.shape)
print("Shape of X_test: ", X_test.shape)


# In[ ]:


shape = X_train.shape


# ## Model cơ bản

# In[20]:


model = Sequential()
model.add(Dense(32, input_shape=(shape[1],), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=10)

y_pred = model.predict_classes(X_test)
print_result(y_pred, y_test, "Deep learning standard: ")


# ## Tăng số lượng tầng Dense
# Kết quả đạt được thì lại kém hơn so với mạng standard cơ bản
# ### Kết luận: Việc tăng số lượng tầng không đem lại hiệu quả

# In[15]:


model2 = Sequential()
model2.add(Dense(32, input_shape=(shape[1],), activation='relu'))
model2.add(Dense(32, activation='relu'))
model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(1, activation='sigmoid'))
model2.summary()

model2.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['acc'])
model2.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=10)

y_pred2 = model2.predict_classes(X_test)
print_result(y_pred2, y_test, "Deep learning standard: ")


# ## Model 3 có độ rộng của tầng Dense tăng lên từ 32 lên 128
# Nhưng kết quả vẫn kém hơn so với model 1
# ### Kết luận: Việc tăng độ rộng tầng Dense không đem lại hiệu quả

# In[19]:


model3 = Sequential()
model3.add(Dense(128, input_shape=(shape[1],), activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(1, activation='sigmoid'))
model3.summary()

model3.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['acc'])
model3.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=10)

y_pred3 = model3.predict_classes(X_test)
print_result(y_pred2, y_test, "Deep learning standard: ")


# In[ ]:




