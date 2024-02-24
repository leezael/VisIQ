import requests
import logging
import json
import configparser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


config = configparser.ConfigParser()
config.readfp(open(r'../config/configuration.yaml'))
envt_url = config.get('DEFAULT', 'envt_url')
username = config.get('DEFAULT', 'username')
password = config.get('DEFAULT', 'password')



#####################################################################################################################################
# HELPS LOGIN TO INTELLIDETECT
#####################################################################################################################################
def login_api(username, password):
    try : 
        api_parms = {
        "username": username,
        "password": password,
        }
        endpoint = f"{envt_url}api/auth/login"
        headers={"Accept":"application/json"}
        response = requests.post(endpoint, json=api_parms,headers=headers)
        if response.ok :
            logger.info("Login Success")
            apikey=response.json().get("apikey")
            return apikey
        elif response.status_code == 401:
            logger.exception("Login Failed : Status Code 401 - Unauthorized: Invalid credentials")
        else:
            logger.exception("Login Failed : Status Code")
    except Exception as e:
        logger.exception("Error : Login Exception - Check login_api call")
        logger.exception(e)

#####################################################################################################################################
# RETRIEVE INFORMATION ABOUT THE USER
#####################################################################################################################################
def get_user(access_token):
    try:
        endpoint = f"{envt_url}/api/users/get"
        headers={"Accept":"application/json","Auth":access_token}
        response = requests.get(endpoint,headers=headers)
        if response.ok :
            logger.info("Get User Success")
            get_user_response=response.json()
        elif response.status_code == 401:
            logger.exception("Get User Failed : Status Code 401 - Unauthorized: Invalid Auth token")
        else:
            logger.exception("Get User Failed : Status Code not 200")
    except Exception as e:
        logger.exception("Error : Get User Exception - Check get_user call")
        logger.exception(e)
    print("User details : ",get_user_response) 
    
#####################################################################################################################################
# RETRIEVE ALL THE USEFUL DATA FOR THE DASHBOARD/HOME PAGE. GIVES THE INFORMATION ON THE COUNT OF MODELS, DATASETS, 
# IMAGES AND INFERENCES.
#####################################################################################################################################
def get_dash(access_token):
    try:
        endpoint = f"{envt_url}/api/dashboard_v2/get"
        headers={"Accept":"application/json","Auth":access_token}
        response = requests.get(endpoint,headers=headers)
        if response.ok :
            logger.info("Get Dash Success")
            get_dash_response=response.json()
        elif response.status_code == 401:
            logger.exception("Get Dash Failed : Status Code 401 - Unauthorized: Invalid Auth token")
        else:
            logger.exception("Get Dash Failed : Status Code not 200")
    except Exception as e:
        logger.exception("Error : Get Dash Exception - Check get_dash call")
        logger.exception(e)
    print("Dashboard details : ",get_dash_response) 

#####################################################################################################################################
# RETRIEVE ALL THE INFORMATION ON THE MODELS THAT ARE ACCESSIBLE TO THE USER
#####################################################################################################################################
def get_model(access_token):
    try:
        endpoint = f"{envt_url}/api/models_v2/get"
        headers={"Accept":"application/json","Auth":access_token}
        response = requests.get(endpoint,headers=headers)
        if response.ok :
            logger.info("Get Model Success")
            get_model_response=response.json()
        elif response.status_code == 401:
            logger.exception("Get Model Failed : Status Code 401 - Unauthorized: Invalid Auth token")
        else:
            logger.exception("Get Model Failed : Status Code not 200")
    except Exception as e:
        logger.exception("Error : Get Model Exception - Check get_model call")
        logger.exception(e)
    print("Model details : ",get_model_response) 

#####################################################################################################################################
# RETRIEVE ALL THE INFORMATION ON THE DATASETS THAT ARE ACCESSIBLE TO THE USER
#####################################################################################################################################
def get_dataset(access_token):
    try:
        endpoint = f"{envt_url}/api/datasets/get"
        headers={"Accept":"application/json","Auth":access_token}
        response = requests.get(endpoint,headers=headers)
        if response.ok :
            logger.info("Get Dataset Success")
            get_dataset_response=response.json()
        elif response.status_code == 401:
            logger.exception("Get Dataset Failed : Status Code 401 - Unauthorized: Invalid Auth token")
        else:
            logger.exception("Get Dataset Failed : Status Code not 200")
    except Exception as e:
        logger.exception("Error : Get Dataset Exception - Check get_model call")
        logger.exception(e)
    print("Dataset details : ",get_dataset_response) 
    


def create_version(access_token,model_id):
    endpoint = f"{envt_url}/api/models_v2/get/{model_id}"
    response = requests.get(endpoint,headers={'Auth':'access_token'} )
    return response

def get_version(access_token):
    endpoint = f"{envt_url}/api/models_v2/get/11"
    response = requests.get(endpoint,headers={'Auth':'access_token'} )
    return response


def create_dataset(access_token,name,description,label,type):
    api_parms = {
        "name": name,
        "description": description,
        "label": label,
        "type": type,
    }
    headers = {}
    headers = {"Accept": "application/json","Auth": f"Bearer {access_token}"}
    endpoint = f"{envt_url}/api/datasets/create"
    response = requests.post(endpoint, params=api_parms, headers=headers).json()
    return response


def upload_file(access_token,model_id):
    api_parms = {

    }
    headers = {}
    headers = {"Accept": "application/json","Auth": f"Bearer {access_token}"}
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
    headers = {"Accept": "application/json","Auth": f"Bearer {access_token}"}
    endpoint = f"{envt_url}/api/models_v2/create"
    response = requests.post(endpoint, params=api_parms, headers=headers).json()
    return response



def inference_image(access_token,version_id):
    api_parms = {

    }
    headers = {}
    headers = {"Accept": "application/json","Auth": f"Bearer {access_token}"}
    endpoint = f"{envt_url}/api/models_v2/inference/{version_id}"
    response = requests.post(endpoint, params=api_parms, headers=headers).json()
    return response


def main(): 
    apikey=login_api(username,password)
    get_user(apikey)
    get_dash(apikey)
    get_model(apikey)
    get_dataset(apikey)
  
if __name__=="__main__": 
    main() 