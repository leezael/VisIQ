import requests
import logging
import json
import configparser
# from requests_toolbelt.utils import dump
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


config = configparser.ConfigParser()
config.readfp(open(r'../config/config.yaml'))
envt_url = config.get('DEFAULT', 'envt_url')
username = config.get('DEFAULT', 'username')
password = config.get('DEFAULT', 'password')


    # classification_models=["resnet18", "vgg16", "alexnet", "googlenet", "mnasnet1_0", "resnet50"] 
    # segmentation_models=["Unet"] 


#####################################################################################################################################
# HELPS LOGIN TO INTELLIDETECT
#####################################################################################################################################
def login_api(username, password):
    print("In login_api")
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
            print(apikey)
            return apikey
        elif response.status_code == 401:
            logger.exception("Login Failed : Status Code 401 - Unauthorized: Invalid credentials")
        else:
            logger.exception(f"Status Code : {response.status_code}")
            logger.exception("Login Failed : Status Code")
    except Exception as e:
        logger.exception("Error : Login Exception - Check login_api call")
        logger.exception(e)

#####################################################################################################################################
# RETRIEVE INFORMATION ABOUT THE USER
#####################################################################################################################################
def get_user(access_token):
    print("In get_user")
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
            logger.exception(f"Status Code : {response.status_code}")
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
    print("In get_dash")
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
            logger.exception(f"Status Code : {response.status_code}")
            logger.exception("Get Dash Failed : Status Code not 200")
    except Exception as e:
        logger.exception("Error : Get Dash Exception - Check get_dash call")
        logger.exception(e)
    print("Dashboard details : ",get_dash_response) 

#####################################################################################################################################
# RETRIEVE ALL THE INFORMATION ON THE MODELS THAT ARE ACCESSIBLE TO THE USER
#####################################################################################################################################
def get_model(access_token):
    print("In get_model")
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
            logger.exception(f"Status Code : {response.status_code}")
            logger.exception("Get Model Failed : Status Code not 200")
    except Exception as e:
        logger.exception("Error : Get Model Exception - Check get_model call")
        logger.exception(e)
    print("Model details : ",get_model_response) 

#####################################################################################################################################
# RETRIEVE ALL THE INFORMATION ON THE DATASETS THAT ARE ACCESSIBLE TO THE USER
#####################################################################################################################################
def get_dataset(access_token):
    print("In get_dataset")
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
            logger.exception(f"Status Code : {response.status_code}")
            logger.exception("Get Dataset Failed : Status Code not 200")
    except Exception as e:
        logger.exception("Error : Get Dataset Exception - Check get_model call")
        logger.exception(e)
    print("Dataset details : ",get_dataset_response) 
    
#####################################################################################################################################
# CREATE A DATASET
#####################################################################################################################################
def create_dataset(access_token,name,description,type,label):
    print("In create_dataset")
    try:
        endpoint = f"{envt_url}/api/datasets/create"
        if type == "Classification":
            api_parms = {
            "name": name,
            "description": description,
            "label": label,
            "type":type,
            }
        else:
            api_parms = {
            "name": name,
            "description": description,
            "type":type,
            }
        headers={"Accept":"application/json","Auth":access_token}
        response = requests.post(endpoint, json=api_parms,headers=headers)
        if response.ok :
            logger.info("Get Dataset Success")
            get_create_dataset_response=response.json()
            print("Created Dataset Information : ",get_create_dataset_response)
        elif response.status_code == 401:
            logger.exception("Create Dataset Failed : Status Code 401 - Unauthorized: Invalid Auth token")
        else:
            logger.exception(f"Status Code : {response.json()}")
            logger.exception("Create Dataset Failed : Status Code not 200")
    except Exception as e:
        logger.exception("Error : Create Dataset Exception - Check create_dataset call")
        logger.exception(e)

#####################################################################################################################################
# UPLOAD A FILE TO THE DATASET
#####################################################################################################################################
def upload_file(access_token,dataset_id,filename):
    print("In upload_file")
    try:
        payload = {}
        files=[
                ('file',(filename,open(filename,'rb'),'application/zip'))
            ]
        headers={"Accept":"application/json","Auth":access_token}
        endpoint = f"{envt_url}api/datasets/upload/{dataset_id}"
        response = requests.request("PUT", endpoint, headers=headers, data=payload, files=files)
        if response.ok :
            logger.info("Upload File Success")
            upload_file_to_dataset_response=response.json()
            print("Uploaded File Information : ",upload_file_to_dataset_response)
        elif response.status_code == 401:
            logger.exception("Upload File Failed : Status Code 401 - Unauthorized: Invalid Auth token")
        else:
            logger.exception(f"Status Code : {response.json()}")
            logger.exception("Upload File Failed : Status Code not 200")
    except Exception as e:
        logger.exception("Error : Upload File Exception - Check upload_file call")
        logger.exception(e)
     
#####################################################################################################################################
# CREATE MODEL USING DATASET IDS
#####################################################################################################################################
def create_model(access_token,name,description,datasets,type):
    print("In create_model")
    try:
        endpoint = f"{envt_url}api/models_v2/create"
        headers={"Content-Type":"application/json","Auth":access_token}
        payload = json.dumps({
        "name": name,
        "description": description,
        "datasets": datasets,
        "type": type,
        })
        response = requests.post(endpoint, headers=headers, data=payload)
        if response.ok :
            logger.info("Create Model Success")
            create_model_response=response.json()
            print("Created Model Information : ",create_model_response)
        elif response.status_code == 401:
            logger.exception("Create Model Failed : Status Code 401 - Unauthorized: Invalid Auth token")
        else:
            logger.exception(f"Status Code : {response.status_code}")
            logger.exception("Create Model Failed : Status Code not 200")
    except Exception as e:
        logger.exception("Error : Create Model Exception - Check create_model call")
        logger.exception(e)

#####################################################################################################################################
# CREATE A NEW VERSION FOR A MODEL USING MODEL ID
#####################################################################################################################################
def create_version(access_token,baseModel,learningRate,momentum,type,epochs,model_id):
    print("In create_version")
    try:
        endpoint = f"{envt_url}api/models_v2/create/{model_id}"
        api_parms = {
            "baseModel": baseModel,
            "learningRate": learningRate,
            "momentum": momentum,
            "type": type,
            "epochs": epochs
        }
        headers={"Accept":"application/json","Auth":access_token}
        response = requests.post(endpoint, json=api_parms,headers=headers)
        if response.ok :
            logger.info("Create Version Success")
            get_version_response=response.json()
            print("response",response)
            print("Created Version Information : ",get_version_response)
        elif response.status_code == 401:
            logger.exception("Create Version Failed : Status Code 401 - Unauthorized: Invalid Auth token")
        else:
            logger.exception(f"Status Code : {response.json()}")
            logger.exception("Create Version Failed : Status Code not 200")
    except Exception as e:
        logger.exception("Error : Create Version Exception - Check create_version call")
        logger.exception(e)

#####################################################################################################################################
# GET ALL VERSION INFORMATION FOR A MODEL USING MODEL ID
#####################################################################################################################################
def get_version(access_token,model_id):
    print("In get_version")
    try:
        endpoint = f"{envt_url}api/models_v2/get/{model_id}"
        headers={"Accept":"application/json","Auth":access_token}
        response = requests.get(endpoint,headers=headers)
        if response.ok :
            logger.info("Get Version Success")
            get_version_response=response.json()
            print("response",response)
            print("Get Version Information : ",get_version_response)
        elif response.status_code == 401:
            logger.exception("Get Version Failed : Status Code 401 - Unauthorized: Invalid Auth token")
        else:
            logger.exception(f"Status Code : {response.json()}")
            logger.exception("Get Version Failed : Status Code not 200")
    except Exception as e:
        logger.exception("Error : Get Version Exception - Check get_version call")
        logger.exception(e)

#####################################################################################################################################
# EXECUTE INFERENCE ON AN IMAGE FOR A MODEL VERSION
#####################################################################################################################################
def make_inference(access_token,version_id,filename,request_id):
    print("In make_inference")
    actual_filename=filename.split("/",-2)[-1]
    
    try:
        payload = {"request_id":request_id}
        files=[
                ('file',(actual_filename,open(filename,'rb'),'image/jpeg'))
            ]

        headers={"Cookie":f"Auth={access_token}"}
        endpoint = f"{envt_url}api/models_v2/inference/{version_id}"
        response = requests.request("PUT", endpoint, headers=headers, data=payload, files=files)

        if response.ok :
            logger.info("Make Inference Success")
            make_inference_response=response.json()[0]
            print("make_inference_response:",make_inference_response)

            return(make_inference_response.get("request_id"),make_inference_response.get("file_name"))
        elif response.status_code == 401:
            logger.exception("Make Inference Failed : Status Code 401 - Unauthorized: Invalid Auth token")
        else:
            logger.exception(f"Status Code : {response.json()}")
            logger.exception("Make Inference Failed : Status Code not 200")
    except Exception as e:
        logger.exception("Error : Make Inference Exception - Check make_inference call")
        logger.exception(e)

#####################################################################################################################################
# GET ALL VERSION INFORMATION FOR A MODEL USING MODEL ID
#####################################################################################################################################
def get_inference_image(access_token,request_id,image_id,output_file_path):
    print("In get_inference_image")
    try:
        endpoint = f"{envt_url}api/results_v2/get/img/{request_id}/{image_id}"
        print("endpoint:",endpoint)
        headers={"Auth":access_token}
        response = requests.get(endpoint,headers=headers,stream=True)
        if response.ok :
            logger.info("Get Inference Success")
            with open(f"{output_file_path}/{image_id}", 'wb') as out_file:
                    shutil.copyfileobj(response.raw, out_file)
            del response

            # get_inference_response=response.json()
            # print("response",response)
            # print("Get Inference Information : ",get_inference_response)
        elif response.status_code == 401:
            logger.exception("Get Inference Failed : Status Code 401 - Unauthorized: Invalid Auth token")
        else:
            logger.exception(f"Status Code : {response.json()}")
            logger.exception("Get Inference Failed : Status Code not 200")
    except Exception as e:
        logger.exception("Error : Get Inference Exception - Check get_inference_image call")
        logger.exception(e)


def format_prepped_request(prepped, encoding=None):
    # prepped has .method, .path_url, .headers and .body attribute to view the request
    encoding = encoding or requests.utils.get_encoding_from_headers(prepped.headers)
    body = prepped.body.decode(encoding) if encoding else '<binary data>' 
    headers = '\n'.join(['{}: {}'.format(*hv) for hv in prepped.headers.items()])
    return f"""{prepped.method} {prepped.path_url} HTTP/1.1{headers}{body}"""

def main(): 

    apikey=login_api(username,password)
    get_user(apikey)
    get_dash(apikey)
    get_model(apikey)
    get_dataset(apikey)

    # type_of_datatset="Classification"
    # dataset_name="SmallDataset2"
    # dataset_description="This is for a trial - SmallDataset2"
    # dataset_label="Tumor"
    # create_dataset(apikey,dataset_name,dataset_description,type_of_datatset,dataset_label)

    # upload_file_dataset_id="929"
    # upload_file_path="../datasets/BrainTumorClassification/Trial/NoTumor.zip"
    # upload_file(apikey,upload_file_dataset_id,upload_file_path)

    # upload_file_dataset_id="930"
    # upload_file_path="../datasets/BrainTumorClassification/Trial/Tumor.zip"
    # upload_file(apikey,upload_file_dataset_id,upload_file_path)

    # string_of_datasets="929,930"
    # list_of_datasets=string_of_datasets.split(",")
    # type_of_model="Classification"
    # model_name="Trial_Model1_classification_tumor"
    # model_description="This is a for trial _ tumor model1"
    # create_model(apikey,model_name,model_description,list_of_datasets,type_of_model)

    # version_baseModel="resnet50"
    # version_learningRate=0.01
    # version_momentum=0.9
    # version_type="Classification"
    # version_epochs=5
    # version_model_id=901
    # create_version(apikey,version_baseModel,version_learningRate,version_momentum,version_type,version_epochs,version_model_id)

    # get_version_model_id=901
    # get_version(apikey,get_version_model_id)

    make_inference_version_id="190"
    make_inference_file_path="../datasets/BrainTumorClassification/Trial/Try1.jpg"
    make_inference_request_id=make_inference_file_path.split("/",-2)[-1].split(".",-1)[0]
    make_inference_request_id,make_inference_file_name=make_inference(apikey,make_inference_version_id,make_inference_file_path,make_inference_request_id)
    print("make_inference_request_id:",make_inference_request_id)
    print("make_inference_file_name:",make_inference_file_name)

    output_file_path="../datasets/BrainTumorClassification/TrialOutput/"
    get_inference_request_id=make_inference_request_id
    get_inference_file_name=make_inference_file_name
    get_inference_image(apikey,get_inference_request_id,get_inference_file_name,output_file_path)
  
if __name__=="__main__": 
    main() 