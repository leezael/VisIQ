import os
import tensorflow as tf
from multiprocessing import Pool
import pandas as pd
import numpy as np
from tqdm import tqdm
import imageio

import seaborn as sns
import matplotlib.pyplot as plt

plt.rc('xtick',labelsize=11)
plt.rc('ytick',labelsize=11)
plt.rcParams["axes.grid"] = False

location_train_dataset_relative='../datasets/chest_xray/train/'


# list_of_chest_categories=os.listdir('../datasets/chest_xray/train/')


def get_image_properties(impath):
    label = impath.split('/')[-2]
    if len(impath.split('.')[0].split('/')[-1].split('-',1)) < 2:
        imgno = impath.split('.')[0].split('/')[-1].split('_',1)[1]
    else:
        imgno = impath.split('.')[0].split('/')[-1].split('-',1)[1]
    
    im = imageio.imread(impath)
    br_med = np.median(im)
    br_std = np.std(im)
    xsize = im.shape[1]
    ysize = im.shape[0]
    datasplit = impath.split('/')[-3]
    return datasplit, impath, label, xsize, ysize, br_med, br_std, imgno



def get_image_properties_all(location_train_dataset_relative):  
    location_train_dataset=os.path.abspath(location_train_dataset_relative)
    list_of_chest_categories = [os.path.join(location_train_dataset, file) for file in os.listdir(location_train_dataset)]
 


    list_of_chest_categories = [item for item in list_of_chest_categories if '.' not in item]
    # print(list_of_chest_categories)
    for directory in list_of_chest_categories:
        print(directory)
        image_paths = tf.io.gfile.glob([directory + '/*.jpeg', ])
        # print(image_paths)
        
        with Pool(4) as p:
            image_props = list(tqdm(p.imap(get_image_properties, image_paths), total=100))

        df = pd.DataFrame(columns=['datasplit', 'path', 'label', 'xsize', 'ysize',
                               'br_med', 'br_std', 'imgno'])
        df['datasplit'] = np.array(image_props).T[0]
        df['path'] = np.array(image_props).T[1]
        df['label'] = np.array(image_props).T[2]
        df['xsize'] = np.array(image_props).T[3].astype(int)
        df['ysize'] = np.array(image_props).T[4].astype(int)
        df['br_med'] = np.array(image_props).T[5].astype(float)
        df['br_std'] = np.array(image_props).T[6].astype(float)
        df['imgno'] = np.array(image_props).T[7]
        df.to_csv('train_image_props.csv', index=False)
        return df

def multivariateGrid(col_x, col_y, col_k, df, k_is_color=False, scatter_alpha=.5, globalhist=False):
    def colored_scatter(x, y, c=None):
        def scatter(*args, **kwargs):
            args = (x, y)
            if c is not None:
                kwargs['c'] = c
            kwargs['alpha'] = scatter_alpha
            plt.scatter(*args, **kwargs)

        return scatter

    g = sns.JointGrid(
        x=col_x,
        y=col_y,
        data=df,
        height=8
    )
    g.set_axis_labels('X Size', 'Y Size')
    
    color = None
    legends=[]
    for name, df_group in df.groupby(col_k):
        legends.append(name)
        if k_is_color:
            color=name
        g.plot_joint(
            colored_scatter(df_group[col_x],df_group[col_y],color),
        )
        sns.distplot(
            df_group[col_x].values,
            ax=g.ax_marg_x,
            color=color,
            kde=False
        )
        sns.distplot(
            df_group[col_y].values,
            ax=g.ax_marg_y,
            color=color,            
            vertical=True,
            kde=False
        )
    # Do also global Hist:
    if globalhist:
        sns.distplot(
            df[col_x].values,
            ax=g.ax_marg_x,
            color='grey'
        )
        sns.distplot(
            df[col_y].values.ravel(),
            ax=g.ax_marg_y,
            color='grey',
            vertical=True
        )
    plt.legend(legends)
    plt.show()
    # plt.savefig('distribution.png')

def main():
    df_image_properties=get_image_properties_all(location_train_dataset_relative)
    df_image_properties.style.set_caption('Head Image Properties')
    multivariateGrid('xsize', 'ysize', 'datasplit', df=df_image_properties)

if __name__ == "__main__":
    main()