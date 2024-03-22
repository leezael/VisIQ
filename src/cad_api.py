import os
import requests
import logging
import json
import configparser
import random
import shutil
import time
import csv


######### classification_models=["resnet18", "vgg16", "alexnet", "googlenet", "mnasnet1_0", "resnet50"] 
######### segmentation_models=["Unet"] 



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

config = configparser.ConfigParser()
config.read_file(open(r'../config/config.yaml'))
envt_url = config.get('DEFAULT', 'envt_url')
username = config.get('DEFAULT', 'username')
password = config.get('DEFAULT', 'password')




#####################################################################################################################################
# How many files to copy to dataset
#####################################################################################################################################
def create_random_sampling(source, dest, no_of_files_for_sample):
    print(f"Creating a sampling of {no_of_files_for_sample} files")
    files = os.listdir(source)
    if not os.path.exists(dest):
        os.makedirs(dest)

    for file_name in random.sample(files, no_of_files_for_sample):
        shutil.copyfile(os.path.join(source, file_name), os.path.join(dest, file_name))


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
            logger.info("Create Dataset Success")
            get_create_dataset_response=response.json()
            print("Created Dataset Information : ",get_create_dataset_response)
            return get_create_dataset_response.get("id")
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

            return(make_inference_response.get("file_name"),make_inference_response.get("label"),make_inference_response.get("confidence"),make_inference_response.get("request_id"),make_inference_response.get("model_id"),make_inference_response.get("model_name"),make_inference_response.get("type"))
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

#####################################################################################################################################
# DELETE DATASET ID
#####################################################################################################################################
def delete_dataset(access_token,dataset_id):
    print("In delete_dataset for : ", dataset_id)
    try:
        endpoint = f"{envt_url}api/datasets/delete/{dataset_id}"
        headers={"Auth":access_token}
        response = requests.delete(endpoint,headers=headers,stream=True)
        if response.ok :
            logger.info("Deleted Dataset ID - Success")
            delete_dataset_response=response.json()
            print("Deleted Dataset Information : ",delete_dataset_response)
        elif response.status_code == 401:
            logger.exception("Delete Dataset Failed : Status Code 401 - Unauthorized: Invalid Auth token")
        else:
            logger.exception(f"Status Code : {response.json()}")
            logger.exception("Delete Dataset Failed : Status Code not 200")
    except Exception as e:
        logger.exception("Error : Delete Dataset Exception - Check delete_dataset call")
        logger.exception(e)


#####################################################################################################################################
# DELETE VERSION ID
#####################################################################################################################################
def delete_version(access_token,version_id):
    print("In delete_version for : ", version_id)
    try:
        endpoint = f"{envt_url}api/models_v2/delete/version/{version_id}"
        headers={"Auth":access_token}
        response = requests.delete(endpoint,headers=headers,stream=True)
        if response.ok :
            logger.info("Deleted Version ID - Success")
            delete_version_response=response.json()
            print("Deleted Version Information : ",delete_version_response)
        elif response.status_code == 401:
            logger.exception("Delete Version Failed : Status Code 401 - Unauthorized: Invalid Auth token")
        else:
            logger.exception(f"Status Code : {response.json()}")
            logger.exception("Delete Version Failed : Status Code not 200")
    except Exception as e:
        logger.exception("Error : Delete Version Exception - Check delete_version call")
        logger.exception(e)

#####################################################################################################################################
# DELETE MODEL ID
#####################################################################################################################################
def delete_model(access_token,model_id):
    print("In delete_model for : ", model_id)
    try:
        endpoint = f"{envt_url}api/models_v2/delete/{model_id}"
        headers={"Auth":access_token}
        response = requests.delete(endpoint,headers=headers,stream=True)
        if response.ok :
            logger.info("Deleted Model ID - Success")
            delete_model_response=response.json()
            print("Deleted Version Information : ",delete_model_response)
        elif response.status_code == 401:
            logger.exception("Delete Model Failed : Status Code 401 - Unauthorized: Invalid Auth token")
        else:
            logger.exception(f"Status Code : {response.json()}")
            logger.exception("Delete Model Failed : Status Code not 200")
    except Exception as e:
        logger.exception("Error : Delete Model Exception - Check delete_model call")
        logger.exception(e)







def main(): 
    apikey=login_api(username,password)
    # get_user(apikey)
    # get_dash(apikey)
    # get_model(apikey)
    # get_dataset(apikey)
    
    
    # # Deletion of Datasets 
    # dataset_to_be_delete=[943,942,930,929,928,927,926,925,924,923,920,919,918]
    # dataset_to_be_delete=[924]
    # for dataset in dataset_to_be_delete:
    #     delete_dataset(apikey,dataset)
    # get_dataset(apikey)


    # # Deletion of Models 
    # model_to_be_delete=[901,900,899,897,924]
    # for model in model_to_be_delete:
    #     delete_model(apikey,model)
    # get_model(apikey)



    # Deletion of versions 
    # version_to_be_delete=[221]
    # for version in version_to_be_delete:
    #     delete_version(apikey,version)
    
    # get_model(apikey)
    



    ############################################################################################################# 
    # Brain Classification
    # Number of files = 500
    # Set 1
    #############################################################################################################
    # type_of_datatset="Classification"
    # # for sizes in [500,1000]:
    # for sizes in [2000]:
    #     for filename in ['glioma','meningioma','pituitary','notumor']:
    #         for i in range(1,4):
                
    #             # source = f'/Users/evp/Documents/GMU/DAEN690/Datasets/Brain_Classification/Combined/Training/{filename}/'         ### Need to be changed based on execution
    #             dest = f'/Users/evp/Documents/GMU/DAEN690/Intellidetect_git/VisIQ/datasets/Brain_Classification/N{sizes}/Set{i}/{filename}' ### Need to be changed based on execution
    #             # create_random_sampling(source, dest,sizes)
    #             # shutil.make_archive(f'{dest}_final', 'zip', dest)

    #             apikey=login_api(username,password)

    #             dataset_name=f"N{sizes}_{filename}_FileSet{i}"
    #             dataset_description=f"Classification_N{sizes}_{filename}_FileSet{i}"
    #             dataset_label=f"{filename}"
    #             dataset_id=create_dataset(apikey,dataset_name,dataset_description,type_of_datatset,dataset_label)

    #             upload_file_dataset_id=dataset_id
    #             upload_file_path=f"{dest}_final.zip"
    #             upload_file(apikey,upload_file_dataset_id,upload_file_path)



    # upload_file_dataset_id="929"
    # upload_file_path="../datasets/BrainTumorClassification/Trial/NoTumor.zip"
    # upload_file(apikey,upload_file_dataset_id,upload_file_path)

    # upload_file_dataset_id="930"
    # upload_file_path="../datasets/BrainTumorClassification/Trial/Tumor.zip"
    # upload_file(apikey,upload_file_dataset_id,upload_file_path)

    

    # string_of_datasets="1044,1047,1050,1053"
    # list_of_datasets=string_of_datasets.split(",")
    # type_of_model="Classification"
    # model_name="N2000_FileSet3_Model"
    # model_description="N2000_FileSet3_Model"
    # create_model(apikey,model_name,model_description,list_of_datasets,type_of_model)
    # get_model(apikey)

    # version_baseModel="resnet18"
    # version_learningRate=0.01
    # version_momentum=0.9
    # version_type="Classification"
    # version_epochs=20
    # version_model_id=912
    # create_version(apikey,version_baseModel,version_learningRate,version_momentum,version_type,version_epochs,version_model_id)

    # version_baseModel="resnet18"
    # version_learningRate=0.01
    # version_momentum=0.9
    # version_type="Classification"
    # version_epochs=50
    # version_model_id=912
    # create_version(apikey,version_baseModel,version_learningRate,version_momentum,version_type,version_epochs,version_model_id)

    # version_baseModel="resnet50"
    # version_learningRate=0.01
    # version_momentum=0.9
    # version_type="Classification"
    # version_epochs=20
    # version_model_id=912
    # create_version(apikey,version_baseModel,version_learningRate,version_momentum,version_type,version_epochs,version_model_id)

    # version_baseModel="resnet50"
    # version_learningRate=0.01
    # version_momentum=0.9
    # version_type="Classification"
    # version_epochs=50
    # version_model_id=912
    # create_version(apikey,version_baseModel,version_learningRate,version_momentum,version_type,version_epochs,version_model_id)

    # version_baseModel="vgg16"
    # version_learningRate=0.01
    # version_momentum=0.9
    # version_type="Classification"
    # version_epochs=20
    # version_model_id=912
    # create_version(apikey,version_baseModel,version_learningRate,version_momentum,version_type,version_epochs,version_model_id)

    # version_baseModel="vgg16"
    # version_learningRate=0.01
    # version_momentum=0.9
    # version_type="Classification"
    # version_epochs=50
    # version_model_id=912
    # create_version(apikey,version_baseModel,version_learningRate,version_momentum,version_type,version_epochs,version_model_id)

    # get_version_model_id=901
    # get_version(apikey,get_version_model_id)

    ##### ------------------------------------------------------------------------------------------------------------------------------------
    ##### ------------------------------------------------------------------------------------------------------------------------------------
    ##### Things to change
    make_inference_version_id=""
    
    dict_model_datasets={"1149":"Positive","1150":"Negative"}
    inference_location="/Users/olungwedc/Desktop/GMU/Spring24/DAEN 690/Avyuct/coronary/dataset/test/"
    #inference_location="/Users/evp/Documents/GMU/DAEN690/Intellidetect_git/VisIQ/datasets/Brain_Classification/Testing_1_and_2/glioma/ALL/"
    # inference_location="/Users/evp/Documents/GMU/DAEN690/Intellidetect_git/VisIQ/datasets/Brain_Classification/Testing_1/notumor/ALL/"
    ##### ------------------------------------------------------------------------------------------------------------------------------------
    ##### ------------------------------------------------------------------------------------------------------------------------------------

    # onlyfiles = [f for f in os.listdir(inference_location) if os.path.isfile(os.path.join(inference_location, f)) and f!=".DS_Store"]
    name1=inference_location.split("/",-1)[-4]
    name2=inference_location.split("/",-1)[-3]
    #name3=inference_location.split("/",-1)[-2]
    key_list = list(dict_model_datasets.keys())
    val_list = list(dict_model_datasets.values())
    actual_dataset= key_list[val_list.index(name2)]
    final_list=[]

    final_file_name=f"FinalDataset{actual_dataset}_Version{make_inference_version_id}_{name1}_{name2}_{name3}.csv"
    onlyfiles = [os.path.join(inference_location, f) for f in os.listdir(inference_location) if os.path.isfile(os.path.join(inference_location, f)) and f!=".DS_Store"]
    for file in onlyfiles:
        file_name=file.split("/",-1)[-1].split(".",-1)[0]
        set_name=file.split("/",-1)[-2]
        actual_class=file.split("/",-1)[-3]
        print(f"Now Processing : FINAL_BrainTumorClassification_ActualDataset{actual_dataset}_Version{make_inference_version_id}_Testing_1_{actual_class}_{set_name}_{file_name}")
        make_inference_file_path=file
        make_inference_request_id=f"FinalDataset{actual_dataset}_Version{make_inference_version_id}_FINAL_BrainTumorClassification_Testing_1_{actual_class}_{set_name}_{file_name}"
        make_inference_file_name,make_inference_label,make_inference_confidence,make_inference_request_id,make_inference_model_id,make_inference_model_name,make_inference_type=make_inference(apikey,make_inference_version_id,make_inference_file_path,make_inference_request_id)
        actual_dataset_name=dict_model_datasets.get(actual_dataset)
        make_inference_label_name=dict_model_datasets.get(make_inference_label)
        if make_inference_label == actual_dataset:
            matched="YES"
        else:
            matched="NO"
        final_string=f"{make_inference_version_id},{actual_dataset},{make_inference_file_name},{make_inference_label},{make_inference_confidence},{make_inference_request_id},{make_inference_model_id},{make_inference_model_name},{make_inference_type},{actual_dataset_name},{make_inference_label_name},{matched}"
        final_string_to_list = list(final_string.split(",")) 
        final_list.append(final_string_to_list)
        print(final_string)
        time.sleep(2)
    
    with open(final_file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(final_list) # Use writerows for nested list


    # make_inference_version_id="190"
    # make_inference_file_path="../datasets/BrainTumorClassification/Trial/Try1.jpg"
    # make_inference_request_id=make_inference_file_path.split("/",-2)[-1].split(".",-1)[0]
    # make_inference_file_name,make_inference_label,make_inference_confidence,make_inference_request_id,make_inference_model_id,make_inference_model_name,make_inference_type=make_inference(apikey,make_inference_version_id,make_inference_file_path,make_inference_request_id)
    # print("make_inference_file_name:",make_inference_file_name)
    # print("make_inference_label:",make_inference_label)
    # print("make_inference_confidence:",make_inference_confidence)
    # print("make_inference_request_id:",make_inference_request_id)
    # print("make_inference_model_id:",make_inference_model_id)
    # print("make_inference_model_name:",make_inference_model_name)
    # print("make_inference_type:",make_inference_type)
    

    # output_file_path="../datasets/BrainTumorClassification/TrialOutput/"
    # get_inference_request_id=make_inference_request_id
    # get_inference_file_name=make_inference_file_name
    # get_inference_image(apikey,get_inference_request_id,get_inference_file_name,output_file_path)
  
if __name__=="__main__": 
    main() 