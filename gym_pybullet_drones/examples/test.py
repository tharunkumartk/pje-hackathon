import requests as req


data = {"direction": "up"}
resp = req.post("http://localhost:5000/control", json=data)

print(resp)
