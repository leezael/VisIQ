import requests

url = "http://54.210.111.216/api/datasets/upload/920"

payload = {}
files=[
  ('file',('NoTumor.zip',open('iEwtB9mL-/NoTumor.zip','rb'),'application/zip'))
]
headers = {
  'Cookie': 'Auth=gAAAAABl250X7wSCp18bT7UmK10g4DPIRhRY5ARzHnYLqJcd5Ul4yo3ojdAMMbRpOKdG41L_fZT5KDMuB9poPZrVNoLHCpjonkn1HkdXbm0ZMqLSFL-Il0KgGeoHa7xgaLmfUZuIMXs8'
}

response = requests.request("PUT", url, headers=headers, data=payload, files=files)

print(response.text)