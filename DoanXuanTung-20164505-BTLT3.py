#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Crypto.Hash import SHA256


# In[2]:


def hashfile(filename):
    with open(filename, 'rb') as fileinput:
        data = fileinput.read()
        
    listbytes = [data[j:j+1024] for j in range(0, len(data), 1024)]
    fileinput.close()
    h = ''
    for i in reversed(range(len(listbytes))):
        F = SHA256.new(listbytes[i] + bytes.fromhex(h))
        h = F.hexdigest()
    return h


# In[3]:


def checkblock(B0, h1, h0):
    # type(B0): bytes 1024 bytes
    # type(h1): hex
    # type(h0): hex
    F = SHA256.new(B0 + bytes.fromhex(h1))
    return (F.hexdigest() == h0)


# In[4]:


print(hashfile('birthday.mp4'))


# In[5]:


with open('birthday.mp4', 'rb') as fileinput:
    data = fileinput.read()
B0 = data[:1024]
fileinput.close()
h0 = '03c08f4ee0b576fe319338139c045c89c3e8e9409633bea29442e21425006ea8'
h1 = 'fa762b85871f31946263fa1d410f417f3442a731dd949355f0266233e25b984b'
print(checkblock(B0, h1, h0))


# In[ ]:




