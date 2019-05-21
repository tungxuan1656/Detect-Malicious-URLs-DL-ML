#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from sklearn.metrics import accuracy_score, f1_score
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Reshape
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
def onehot_coding_data(data, char_dict):
    # convert
    data = [[char_dict[el] for el in line] for line in data]

    # set max len element of data
    for i in range(len(data)):
        if (len(data[i]) < 300):
            data[i] = data[i] + [0]*(300 - len(data[i]))
        else:
            data[i] = data[i][:300]

    # one-hot vector
    X = np.asarray([to_categorical(i, num_classes=63) for i in data])

    return X


# In[ ]:


bad_requests = loadData('normalRequest.txt')
good_requests = loadData('anomalousRequest.txt')


# In[ ]:


all_requests = bad_requests + good_requests
y_bad = [1] * len(bad_requests)
y_good = [0] * len(good_requests)


# In[5]:


print("Total requests : ", len(all_requests))
print("Bad requests: ", len(bad_requests))
print("Good requests: ", len(good_requests))


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


# In[8]:


print("Requests for Train data: (%d); (normal, malicious) = (%d, %d)"%(len(train), len(normal_train), len(malicious_train)))
print("Requests for Validation data: (%d); (normal, malicious) = (%d, %d)"%(len(val), len(normal_val), len(malicious_val)))
print("Requests for Test data: (%d); (normal, malicious) = (%d, %d)"%(len(test), len(normal_test), len(malicious_test)))
print("Use Trigram (n=3). Split Train:Validation:Test = 6:2:2\n")


# In[ ]:


# create dict
char_dict = {}
char_smpl = ' '.join(train)
char_smpl = sorted(list(set(char_smpl)))
for idx, ch in enumerate(char_smpl):
    char_dict[ch] = idx


# In[ ]:


X_train = onehot_coding_data(train, char_dict)
X_val = onehot_coding_data(val, char_dict)
X_test = onehot_coding_data(test, char_dict)


# In[11]:


shape = X_train.shape
print("Shape of X_train: ", X_train.shape)
print("Shape of X_val: ", X_val.shape)
print("Shape of X_test: ", X_test.shape)


# ## Mạng CNN cơ bản với 3 tầng Conv2D và độ rộng là 32

# In[12]:


model = Sequential()
model.add(Reshape((shape[1], shape[2], 1), input_shape=(shape[1], shape[2])))

model.add(Conv2D(32, (3, 63), activation='relu'))
model.add(Conv2D(32, (3, 1), activation='relu'))
model.add(Conv2D(32, (3, 1), activation='relu'))

model.add(Dropout(0.2))
model.add(GlobalMaxPooling2D())

model.add(Dense(64))
model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_val, y_val))

y_pred = model.predict_classes(X_test)
print_result(y_pred, y_test, 'CNN Conv2d: ')


# ## Model 2 so với model 1 có 5 lớp Conv2D
# Nhưng kết quả đạt được không hiệu quả hơn so với model 1
# ### Kết luận: Số tầng Conv2D đến một mức nào đó đã là đạt hiệu quả cao nhất, tăng thêm cũng không đạt được hiệu quả nữa

# In[13]:


model2 = Sequential()
model2.add(Reshape((shape[1], shape[2], 1), input_shape=(shape[1], shape[2])))

model2.add(Conv2D(32, (3, 63), activation='relu'))
model2.add(Conv2D(32, (3, 1), activation='relu'))
model2.add(Conv2D(32, (3, 1), activation='relu'))
model2.add(Conv2D(32, (3, 1), activation='relu'))
model2.add(Conv2D(32, (3, 1), activation='relu'))

model2.add(Dropout(0.2))
model2.add(GlobalMaxPooling2D())

model2.add(Dense(64))
model2.add(Dense(1, activation='sigmoid'))

model2.summary()
model2.compile(loss='binary_crossentropy',
               optimizer='adam',
               metrics=['acc'])

model2.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_val, y_val))

y_pred2 = model2.predict_classes(X_test)
print_result(y_pred2, y_test, 'CNN Conv2d: ')


# ## Model 3 không tăng số tầng Conv2D mà tăng lên độ rộng của nó
# Kết quả đạt được đã cho thấy hiệu quả được tăng lên khi tăng độ rộng các tầng
# ### Kết luận: Tăng độ rộng từng tầng Conv2D có khả năng nâng cao hiệu quả của mô hình

# In[16]:


model3 = Sequential()

model3.add(Reshape((shape[1], shape[2], 1), input_shape=(shape[1], shape[2])))

model3.add(Conv2D(128, (3, 63), activation='relu'))
model3.add(Conv2D(128, (3, 1), activation='relu'))
model3.add(Conv2D(128, (3, 1), activation='relu'))

model3.add(Dropout(0.2))
model3.add(GlobalMaxPooling2D())

model3.add(Dense(128))
model3.add(Dense(1, activation='sigmoid'))

model3.summary()
model3.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model3.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_val, y_val))

y_pred3 = model3.predict_classes(X_test)
print_result(y_pred3, y_test, 'CNN Conv2d: ')


# ## Model 4 kết hợp model 2 và 3, vừa tăng số tầng Conv2D vừa tăng độ rộng của nó
# Kết quả đạt được đã không được cải thiện thêm

# In[15]:


model4 = Sequential()

model4.add(Reshape((shape[1], shape[2], 1), input_shape=(shape[1], shape[2])))

model4.add(Conv2D(128, (3, 63), activation='relu'))
model4.add(Conv2D(128, (3, 1), activation='relu'))
model4.add(Conv2D(128, (3, 1), activation='relu'))
model4.add(Conv2D(128, (3, 1), activation='relu'))
model4.add(Conv2D(128, (3, 1), activation='relu'))

model4.add(Dropout(0.2))
model4.add(GlobalMaxPooling2D())

model4.add(Dense(128))
model4.add(Dense(1, activation='sigmoid'))

model4.summary()
model4.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model4.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_val, y_val))

y_pred4 = model4.predict_classes(X_test)
print_result(y_pred4, y_test, 'CNN Conv2d: ')


# In[ ]:




