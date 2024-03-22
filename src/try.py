import requests
import json

url = "http://54.210.111.216/api/models_v2/create"

payload = json.dumps({"name": "Test Model25", "description": "This is a for trial model", "datasets": [926, 928], "type": "Classification"})
headers={'Content-Type': 'application/json', 'Auth': 'gAAAAABl27wSUVpYTW5-aVva_-GACVVx3OeNCzHkWLt4y0BFJc_CSgEO2VrtUmXy0A7TtG8H0VttZDy-EOA811ej3dk7BKAMU1npXDX6Gx7C5SVzhHGhMVfGH1-6A3yNcr5UQCYFs1XB'}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
print(response.status_code)
