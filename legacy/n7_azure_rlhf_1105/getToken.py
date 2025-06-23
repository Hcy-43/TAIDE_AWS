import os
import requests
host = "https://td.nchc.org.tw/api/v1"
username = input("username: ")
password = input("password: ")
#get token
r = requests.post(host+"/token", data={"username":username,
"password":password})

print(r.json())

token = r.json()["access_token"]
print(token)