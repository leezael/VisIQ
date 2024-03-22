import os
def main():
    starter=0
    folder = "../datasets/Brain_Classification/Testing_1/glioma"
    for count, filename in enumerate(os.listdir(folder)):
        filecount=starter+count
        dst = f"glioma_set_{str(filecount)}.jpg"
        src = f"{folder}/{filename}"  # foldername/filename, if .jpg file is outside folder
        dst = f"{folder}/{dst}"

        os.rename(src, dst)


# Driver Code
if __name__ == '__main__':
    main()