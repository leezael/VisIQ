import requests

envt_url=""



def login_api(user, pwd):
    api_parms = {
        "username": user,
        "password": pwd,
    }
    headers = {"Accept": "application/json"}
    endpoint = f"{envt_url}/api/users/get"
    response = requests.post(endpoint, params=api_parms, headers=headers).json()
    return response


def get_user(access_token):
    endpoint = f"{envt_url}/api/users/get"
    response = requests.get(endpoint,headers={'Authorization':'access_token'} )
    return response



def get_dash(access_token):
    endpoint = f"{envt_url}/api/dashboard_v2/get"
    response = requests.get(endpoint,headers={'Authorization':'access_token'} )
    return response


def get_model(access_token):
    endpoint = f"{envt_url}/api/models_v2/get"
    response = requests.get(endpoint,headers={'Authorization':'access_token'} )
    return response

def get_dataset(access_token):
    endpoint = f"{envt_url}/api/datasets/get"
    response = requests.get(endpoint,headers={'Authorization':'access_token'} )
    return response

def create_version(access_token,model_id):
    endpoint = f"{envt_url}/api/models_v2/get/{model_id}"
    response = requests.get(endpoint,headers={'Authorization':'access_token'} )
    return response

def get_version(access_token):
    endpoint = f"{envt_url}/api/models_v2/get/11"
    response = requests.get(endpoint,headers={'Authorization':'access_token'} )
    return response


def create_dataset(access_token,name,description,label,type):
    api_parms = {
        "name": name,
        "description": description,
        "label": label,
        "type": type,
    }
    headers = {}
    headers = {"Accept": "application/json","Authorization": f"Bearer {access_token}"}
    endpoint = f"{envt_url}/api/datasets/create"
    response = requests.post(endpoint, params=api_parms, headers=headers).json()
    return response


def upload_file(access_token,model_id):
    api_parms = {

    }
    headers = {}
    headers = {"Accept": "application/json","Authorization": f"Bearer {access_token}"}
    endpoint = f"{envt_url}/api/datasets/upload/{model_id}"
    response = requests.post(endpoint, params=api_parms, headers=headers).json()
    return response

def create_model(access_token,name,description,datasets,type):
    api_parms = {
        "name": name,
        "description": description,
        "datasets": datasets,
        "type": type,
    }
    headers = {}
    headers = {"Accept": "application/json","Authorization": f"Bearer {access_token}"}
    endpoint = f"{envt_url}/api/models_v2/create"
    response = requests.post(endpoint, params=api_parms, headers=headers).json()
    return response



def inference_image(access_token,version_id):
    api_parms = {

    }
    headers = {}
    headers = {"Accept": "application/json","Authorization": f"Bearer {access_token}"}
    endpoint = f"{envt_url}/api/models_v2/inference/{version_id}"
    response = requests.post(endpoint, params=api_parms, headers=headers).json()
    return response


def main(): 
    print("Intellidetect") 
  
if __name__=="__main__": 
    main() 