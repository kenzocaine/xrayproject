import requests

#SERVER_URL = 'http://0.0.0.0:8000'
SERVER_URL = 'https://xray-vx3dknw5ea-ew.a.run.app'
f = open("/Users/kimhedelin/code/kenzocaine/xrayproject/raw_data/ChinaSet_AllFiles/CXR_png/CHNCXR_0619_1.png", 'rb')
files = {"file": (f.name, f, "multipart/form-data")}
o = requests.post(url=f"{SERVER_URL}/files", files=files)
print(o.content)

