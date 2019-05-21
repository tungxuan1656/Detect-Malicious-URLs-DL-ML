#!/usr/bin/env python
# coding: utf-8

# In[1]:


import urllib.parse
import os
import io


# In[2]:


def parse_file(file_in, file_out):
    fin = open(file_in)
    fout = io.open(file_out, "w", encoding="utf-8")
    lines = fin.readlines()
    res = []
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith("GET"):
            res.append("GET" + line.split(" ")[1])
        elif line.startswith("POST") or line.startswith("PUT"):
            url = line.split(' ')[0] + line.split(' ')[1]
            j = 1
            while True:
                if lines[i + j].startswith("Content-Length"):
                    break
                j += 1
            j += 1
            data = lines[i + j + 1].strip()
            url += '?' + data
            res.append(url)
    for line in res:
        line = urllib.parse.unquote(line).replace('\n', '').lower()
        fout.writelines(line + '\n')
    print("finished parse ", len(res), " requests")
    fout.close()
    fin.close()


# In[3]:


normal_file_raw = 'normalTrafficTraining.txt'
anomaly_file_raw = 'anomalousTrafficTest.txt'

normal_file_parse = 'normalRequest.txt'
anomaly_file_parse = 'anomalousRequest.txt'


# In[4]:


# if not os.path.exists('anomalousRequest.txt') or not os.path.exists('normalRequest.txt'):
    parse_file(normal_file_raw, normal_file_parse)
    parse_file(anomaly_file_raw, anomaly_file_parse)


# In[ ]:




