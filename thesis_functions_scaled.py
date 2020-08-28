# -*- coding: utf-8 -*-
"""
Created on Monday 18 may 2020

All the thesis code, no code excecution!
@author: Dainean
"""

#Prepare the python system 
import pandas as pd              #Dataframes
import numpy as np               #Numpy

# Reading and saving fits files
import os                        #Move around in our OS
from astropy.table import Table
from astropy.io import fits  #Working with fits

#Isolation Foreststuffs
import eif as iso                #Expanded Isolation Forest

#Clustering
from scipy.sparse import diags  # Laplacian scoring
from skfeature.utility.construct_W import construct_W  # Laplacian scoring
from sklearn.cluster import KMeans   #Kmeans clustering
from sklearn.preprocessing import StandardScaler

# For PFA
from sklearn.decomposition import PCA
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances


# Plotting
import matplotlib.pyplot as plt
import seaborn as sns                #improved plots



#Working directory control
cwd = os.getcwd() 

#Selecting dataset
#change dataset here, Alpha, prichi or beta 
#dataset = "Alpha"      #Initial max row dataset
#dataset = "prichi"     #prichi < 3 filtered dataset, 24999 rows. OBSELETE
#dataset = "beta"       #prichi < 2 filtered dataset, 13787 rows
#dataset = "gamma"      #prichi < 2 filtered dataset, (removed photometric)) OBSELETE
#dataset = "delta"      #updated DB creator, based on GaussFitSimple, 28128  rows
#dataset = "epsilon"    #trimmed down version of delta, prichi <2, 10941 rows (for easier computation)

#dataset = "zeta"       # Full Photometric, GaussFitSimple, prichi <2, 10941 rows × 134 columns
#dataset = "zeta"       # Full Photometric, GaussFitSimple, prichi <2, 10941 rows × 134 columns
dataset = "eta"       # Full Photometric, GaussFitSimple, all columns

detect_path = True   #this is for easier working in spyder

 #Set up directory path, load initial dataframes
if detect_path == True:
    print("Initial working directory is:", cwd) 
    if '31618' in cwd:
        print("Working at Dora")
        location = "dora"
    if 'Dainean' in cwd:
        print("Working at home, changing to onedrive folder")
        location = "home"
    if 'Onedrive' in cwd:
        print("Working in onedrive folder")
        location = "home"
    if 'Dropbox' in cwd:
        print("Working at home, changing to onedrive folder")
        location = "home"
    
    
    if location == "home":
        os.chdir('D:\Onedrive\Thesis\support\%s'%(dataset))
        print(os.getcwd())

    
    if location == "dora":
        os.chdir('C:\Sander\support\%s'%(dataset))
        print(os.getcwd())
            
    #Loading dataframes     Only part for now
    phot = pd.read_hdf('Parts_DB.h5', 'Photometric') 
    col = pd.read_hdf('Parts_DB.h5', 'Colour')  
    spec = pd.read_hdf('Parts_DB.h5', 'Spectral') 
    
    full = pd.read_hdf('ThesisDB.h5', 'Dataframe') 
    
    dropped = int(phot.shape[0] * 0.05)   #we can safely drop 5% of our dataset. 
    # Is this enough with such a large feature space? It seems to be more then we get by filtering EIF above 0.5 out!

#full = full.iloc[:,6:] #Addition
    combi = pd.merge(phot,spec, right_index=True, left_index=True, how='inner')  #just phot and spec


#full[full.columns[full.columns.str.endswith('u')]]
#a = np.array(['size90u', 'ABSMAGu','MU@Eu','HA_EW', 'OII_EW'])
#inv = full[a]
#often used to return the name of a dataframe as a string

# Assorted functions: 
def get_df_name(df):
    """returns the name of a dataframe as a string"""
    name =[x for x in globals() if globals()[x] is df][0]
    return name

def pandafy(fits_filename):
    """
    Turns an .fits file into a pandas dataframe"""
    dat = Table.read(fits_filename, format='fits')
    df = dat.to_pandas(index = 'CATAID')
    return(df)

def contains(df, string):
    df = df[df.columns[df.columns.str.contains(string)]]
    return df
    
def endswith(df, string):
    df = df[df.columns[df.columns.str.endswith(string)]]
    return df

def startswith(df, string):
    df = df[df.columns[df.columns.str.startswith(string)]]
    return df
    
def fittify(df,filename='ThesisDB_selected.fits'):  #say which dataframe you want to turn into a fit file
    holder = []
    for i in range(df.columns.values.size):
        holder.append(fits.Column(name=df.columns.values[i], format='D', array=df.iloc[:,i]))

    cols = fits.ColDefs(holder)
    hdu = fits.BinTableHDU.from_columns(cols)

    hdu.writeto(filename,overwrite=True)
#%%  EIF Isolation
# Removes the most isolated points from a dataframe using EIF
def eif_isolation(pd_df,dropped = 500,ntrees=1024,sample_size=512,remake=False,save = True):
    """
    Removes the most isolated points from a DataFrame using EIF
    -------------------------------
    Input:
    pd_df: pandas dataframe
    dropped: how many values to drop afterwards
    ntrees: how many trees to make for EIF
    sample_size: how many samples to initiate EIF with
    remake: wether or not to remake if results are found
    save: save the results (needs to be disabled for certain recursions)
    --------------------
    proces:
    Removes the dropped most isolated points from a DataFrame using EIF
    --------------
    Retuns:   
    Returns: New dataframe, where the least relevant datapoints have been dropped
    """
    
    #Set up variables 
    try:
        df_name = get_df_name(pd_df)
    except (IndexError):
        df_name = pd_df.name
        
    while True:
        try:
            if remake == True:
                print("New file requested")
                raise NameError('remake')
            df_isolated = pd.read_hdf('eif_results.h5',"_%s_%i_dropped_%i_%i"\
                                      %(df_name,dropped,ntrees,sample_size))
            print("succes, EIF sorted matrix found")
            print("settings: Dataframe = %s, number dropped = %i, number of trees = %i, samplesize = %i"\
                  %(df_name,dropped,ntrees,sample_size))
            break
        except (FileNotFoundError,KeyError,NameError):
            print("Failed to find this combination, creating one")
            # main bit of code goes here: 
            values = pd_df.values.astype('double')    # numpy array. .astype('double') is as spec is in float32 while EIF expects float64
            elevel = (values.shape[1]-1)  #only doing one extension level anymore, but the largest
            
            EIF_model  = iso.iForest(values, ntrees=ntrees, sample_size=sample_size, ExtensionLevel=elevel) #create a model
            EIF_paths  = EIF_model.compute_paths(X_in=values)  #calculate isolation value for every point
            EIF_sorted = np.argsort(EIF_paths)     #sort these by integers from least to most isolated 
            
            np_remainder = values[:][EIF_sorted[0:-dropped]]       #drop values
            index = pd_df.index.values[:][EIF_sorted[0:-(dropped)]]   #Create a new index that has the same ordering (CATAID)
            
            df_isolated = pd.DataFrame(np_remainder, columns = pd_df.columns.values, index = index)  #selected dataframe
            if save == True:
                df_isolated.to_hdf('eif_results.h5',"_%s_%i_dropped_%i_%i"%(df_name,dropped,ntrees,sample_size))
            print('EIF sorted matrix created and saved')
            print("settings: Dataframe = %s, number dropped = %i, number of trees = %i, samplesize = %i"%(df_name,dropped,ntrees,sample_size))
            break
    return df_isolated



    
    
#setup filtered dataframes
remake = False
phot_eif  = eif_isolation(phot, dropped = dropped, remake = remake)
phot_eif.name = 'Photometric'
spec_eif  = eif_isolation(spec, dropped = dropped, remake = remake)
spec_eif.name = 'Spectral'
combi_eif = eif_isolation(combi, dropped = dropped, remake = remake)
combi_eif.name =  'Combined'
#%%  
remake = False
# dataframe around u 
u_df =  full[full.columns[full.columns.str.endswith('u')]]
u_df.name = "u_phot"
u_eif = eif_isolation(u_df, dropped = dropped, remake = remake)
u_eif.name = 'u_phot'

# dataframe around g 
g_df =  full[full.columns[full.columns.str.endswith('g')]]
g_df.name = "g_phot"
g_eif = eif_isolation(u_df, dropped = dropped, remake = remake)
g_eif.name = 'g_phot'

# dataframe around r
r_df =  full[full.columns[full.columns.str.endswith('r')]]
r_df.name = "r_phot"
r_eif = eif_isolation(u_df, dropped = dropped, remake = remake)
r_eif.name = 'r_phot'


# sample if we want really quick testing
sample = phot_eif.sample(1000)

dataframes = [phot_eif,spec_eif,combi_eif]
k_list = [2,3,4]

#inv_eif =  eif_isolation(inv, dropped = dropped, remake = False)
#inv_eif.name = "investigate"

#inv_eif2 =  eif_isolation(inv, dropped = dropped*2, remake = False)
#inv_eif2.name = "investigateplus"

"""
col_eif   = eif_isolation(col, dropped = dropped, remake = remake)
spec_eif  = eif_isolation(spec, dropped = dropped, remake = remake)
full_eif  = eif_isolation(full, dropped = dropped, remake = remake)
"""
#%%  
# 2 d heatmap for EIF
def getVals(forest,x,sorted=True):
    theta = np.linspace(0,2*np.pi, forest.ntrees)
    r = []
    for i in range(forest.ntrees):
        temp = forest.compute_paths_single_tree(np.array([x]),i)
        r.append(temp[0])
    if sorted:
        r = np.sort(np.array(r))
    return r, theta

def fmax(x):
    if x.max() > 0:
        xmax = x.max()*1.1
    else:
        xmax = x.max()*0.9
    return xmax

def fmin(x):
    if x.min() > 0:
        xmin = x.min()*0.9
    else:
        xmin = x.min()*1.1
    return xmin


def heat_plot(i=6,j=18,df = phot):
    """
    Plots Anomaly score contour for iForest and EIF

    Parameters
    ----------
    i : Integer,
        First column of the dataframe to use. The default is 6.
    j : Integer
        First column of the dataframe to use. The default is 18.
    df : pandas dataframe
       Pandas dataframe to compare The default is phot.

    Returns
    -------
    Created anomaly score contour plots
    """
 


    ntrees = 512      #number of trees we use
    sample_size=512   #how many data points we sample to create our forest
    grid_density = 60   #Density of the grid we make

    iname = df.columns[i]
    jname = df.columns[j]

    #define x and y (easier later)
    np_array = df.values   # converts df into numpy object
    np_array = np_array.astype('double')  #Type is sometimes confused. Easiest to just force
    x, y = np_array[:,i], np_array[:,j]
    bigX = np.array([x,y]).T      #combine them into a single object
      
    # grabbing a 2d plane from the bigger datafield
    #Sample to calculate over in 2d plane

    xx, yy = np.meshgrid(np.linspace(fmin(x), fmax(x), grid_density), 
                         np.linspace(fmin(y), fmax(y), grid_density))


    elevel = [0,1]  #0 is normal IF, 1 is EIF
    counter = 0


    for k in elevel:
        #Calculations
        counter += 1
        F0  = iso.iForest(bigX, ntrees=ntrees, sample_size=sample_size, ExtensionLevel=k) 
        grid = F0.compute_paths(X_in=np.c_[xx.ravel(), yy.ravel()])
        grid = grid.reshape(xx.shape)

        #plotting
        f = plt.figure(figsize=(10,8))
        ax1 = f.add_subplot()
        levels = np.linspace(np.min(grid),np.max(grid),20)
        CS = ax1.contourf(xx, yy, grid, levels, cmap=plt.cm.OrRd)  #alt colour = cmap=plt.cm.YlOrRd)  #alt colour = plt.cm.Blues_r
        plt.scatter(x[::2],y[::2],s=1.8,c='k',edgecolor='None')

        rn, thetan = getVals(F0,np.array([10.,0.]),sorted=sorted)
        ra, thetaa = getVals(F0,np.array([0.,0.]),sorted=sorted)

        if counter == 1:
            ax1.set_title("Generic Isolation Forest\nNominal: Mean={0:.3f}, Var={1:.3f}\nAnomaly: Mean={2:.3f}, Var={3:.3f}".
                          format(np.mean(rn),np.var(rn),np.mean(ra),np.var(ra)))
        else:
            ax1.set_title("Extended Isolation Forest\nNominal: Mean={0:.3f}, Var={1:.3f}\nAnomaly: Mean={2:.3f}, Var={3:.3f}".
                          format(np.mean(rn),np.var(rn),np.mean(ra),np.var(ra)))
        ax1.set_xlabel("%s" %(iname), fontsize=14)
        ax1.set_ylabel("%s" %(jname), fontsize=14)
        cbar = ax1.figure.colorbar(CS)   #colorbar
        f.savefig("pics/eif/IF for %s vs %s, elevel = %i.png" %(iname,jname,k),bbox_inches="tight")
        plt.show()
#heat_plot(9,0,phot)  #alpha
#heat_plot(5,0,phot)  #Epsilon 
        
#%% 

def eif_plot(df,dropped = 500,ntrees=1024,sample_size=512):
    values = df.values.astype('double')    # numpy array. .astype('double') is as spec is in float32 while EIF expects float64
    elevel = (values.shape[1]-1)  #only doing one ext
    
    EIF_model  = iso.iForest(values, ntrees=ntrees, sample_size=sample_size, ExtensionLevel=elevel) #create a model
    EIF_paths  = EIF_model.compute_paths(X_in=values)  #calculate isolation value for every point
    #EIF_sorted = np.argsort(EIF_paths)     #sort these by integers from least to most isolated 
  #  IR = EIF_paths[EIF_paths > 0.6]
    bins = np.linspace(0,1,11)
    plt.hist(EIF_paths,bins=bins, log=True)
    plt.show()
#eif_plot(phot)
#%% 

    
# 2d comparison

def Elevel_plot(df, i=1,j=18,isolated=.6):
    """
    Parameters
    ----------
    df : Pandas Dataframe
        For which pandaframe do we plot a comparison by elevel?
    i : integer, optional
        which column do we select for x, default = 1.
    j : integer, optional
        which column do we select for x, default = 18.
    isolated : float, optional
        Minimum isolation value to count as anomaly
        Number of points to isolate. The default is 500.

    Returns
    -------
    None.

    Plots for 3 different expansion levels an x vs y diagram, so that differences in
    expansion level with regards to the Expanded Isolation Forest (EIF) can be made clear.
    Pulls data from earlier calculations in the holdit_name.npy form. If these are not present,
    they need to be remade first
    """
    sns.set_style("darkgrid")
       
    #  a = holdit(name)  #argsort of our findings
        
        
    iname = df.columns[i]
    jname = df.columns[j]
    
    # ------------------------------------
    # Select a subset of the data
    values = df.values.astype('double')

    iname = df.columns[i]       
    jname = df.columns[j]
    ntrees= 512
    sample_size = 512

    max_extension = (values.shape[1]-1)       #maximum extension level
    elevel = [0,1,max_extension//2,max_extension]  #extension levels

    dot_size = 5
    counter = 0
    

    f = plt.figure(figsize=(18,18))
    for k in elevel:
        EIF_model  = iso.iForest(values, ntrees=ntrees, sample_size=sample_size, ExtensionLevel=k) #create a model
        EIF_paths  = EIF_model.compute_paths(X_in=values)  #calculate isolation value for every point
        EIF_sorted = np.argsort(EIF_paths)     #sort these by integers from least to most isolated 
        
        ordered = EIF_paths[EIF_sorted]
        dropped = ordered[ordered > isolated].size  # Turn into index of how many to drop
        
        np_remainder = values[:][EIF_sorted[:-dropped]]       #remaining values
        np_dropped = values[:][EIF_sorted[-dropped:]]       # dropped values

        #plotting
        ax = f.add_subplot(2,2,counter+1)
        ax.scatter(np_remainder[:,i],np_remainder[:,j],s=dot_size,c='k',alpha = 0.3, edgecolor='none',label="Normal datapoints")
        ax.scatter(np_dropped[:,i],np_dropped[:,j],s=dot_size,c='red',edgecolor='none',label="Anomalous points, s>%.2f"%(isolated))
        # ax.scatter(train1[:,i][a[:isolated,counter]],train1[:,j][a[:isolated,counter]],s=dot_size,c='None',edgecolor='r',label="most central")
        counter += 1
        #    plt.axis("equal")
        plt.title('%i most isolated :Extension level %i'%(dropped,k))
        plt.xlabel(iname)
        plt.ylabel(jname)
        plt.legend(loc="upper left")
    f.savefig("pics/EIF/Comparison '%s vs %s %i isolated"%(iname,jname,dropped),bbox_inches="tight")# 2d comparison
    plt.show()
#Elevel_plot(phot,1,17,0.5)
#Elevel_plot(phot,12,26,0.5)
#%%


#cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)

teller = 0

def cormat(dataframe,annotstate=False):
    global teller
    corrmat = dataframe.corr()
    #top_corr_features = corrmat.index
    plt.figure(figsize=(12,10))
    #plot heat map
    sns.heatmap(corrmat,annot=annotstate, vmin = -1, vmax = 1, center = 0, cmap='coolwarm')
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.title("Dataset: %s, %i objects "%(dataset, dataframe.shape[0]),size=14)
    plt.savefig("pics/cormat_%i"%(teller),bbox_inches="tight")
    teller += 1
    plt.show() 



#determine the laplacian score
def lap_score(X, **kwargs):
    """
    This function implements the laplacian score feature selection, steps are as follows:
    1. Construct the affinity matrix W if it is not specified
    2. For the r-th feature, we define fr = X(:,r), D = diag(W*ones), ones = [1,...,1]', L = D - W
    3. Let fr_hat = fr - (fr'*D*ones)*ones/(ones'*D*ones)
    4. Laplacian score for the r-th feature is score = (fr_hat'*L*fr_hat)/(fr_hat'*D*fr_hat)
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    kwargs: {dictionary}
        W: {sparse matrix}, shape (n_samples, n_samples)
            input affinity matrix
    Output
    ------
    score: {numpy array}, shape (n_features,)
        laplacian score for each feature
    Reference
    ---------
    He, Xiaofei et al. "Laplacian Score for Feature Selection." NIPS 2005.
    """

    # if 'W' is not specified, use the default W
    if 'W' not in kwargs.keys():
        W = construct_W(X)
    # construct the affinity matrix W
    else: W = kwargs['W']    #there was a bug here, fixed!!!!!!!!!!!!!!!!!!
    # build the diagonal D matrix from affinity matrix W
    D = np.array(W.sum(axis=1))
    L = W
    tmp = np.dot(np.transpose(D), X)
    D = diags(np.transpose(D), [0])
    Xt = np.transpose(X)
    t1 = np.transpose(np.dot(Xt, D.todense()))
    t2 = np.transpose(np.dot(Xt, L.todense()))
    # compute the numerator of Lr
    D_prime = np.sum(np.multiply(t1, X), 0) - np.multiply(tmp, tmp)/D.sum()
    # compute the denominator of Lr
    L_prime = np.sum(np.multiply(t2, X), 0) - np.multiply(tmp, tmp)/D.sum()
    # avoid the denominator of Lr to be 0
    D_prime[D_prime < 1e-12] = 10000

    # compute laplacian score for all features
    score = 1 - np.array(np.multiply(L_prime, 1/D_prime))[0, :]
    return np.transpose(score)
 #%%   
#determine the ordering of features based on lap score
def laplace_order(eif_df, remake=False, save = True):
    """
    input: 
        eif_df: a pandas dataframe that already has eif_isolation applied
        remake: wether or not to remake the laplace dataframe, even if it already exists:
        save: save the database? (Turn this off when using recursion)
    ---------------
    method:
        applies lapscore algorhitm (with standard kwargs for standard W) and sorts it
    -------------
    returns:
        Dataframe with column (feature) names, laplacian score, and the number of the related column
    """
    
    if save == True:  #get the name for saving
        try:
            DF_name = eif_df.name
        except:
            DF_name = "custom"

    
    while True:
        try:
            if remake == True:
                print("New laplacian file requested")
                raise NameError('Remake')
            results =  pd.read_hdf('lap_results.h5',"lap_order_%s_dataset_%s_dataframe_%i_filtered"%(dataset,DF_name, dropped))
            print("succes, Laplacian results found")
            print("settings: Dataset = %s, Dataframe = %s, filtered by EIF = %i "%(dataset,DF_name,dropped))
            break
        except (KeyError,FileNotFoundError, NameError):
            print("Failed to find Laplacian results, creating database")
            array = eif_df.values.astype("float32") # Turn df into a 32bit numpy array (to save on memory)
            lapscore = lap_score(array)   #determine the laplacian score of every element
            
            sort = np.sort(lapscore,0)
            ranked = np.argsort(lapscore,0)   #rank features by laplacian scores. Lower means more important
            c_names = eif_df.columns[ranked]  #return the names of the columns these features belong to
           
            # Turn this into a dataframe
            data = {'feature':c_names,'laplacian_score':sort, 'column_nr' : ranked}
            results = pd.DataFrame(data)  #selected dataframe

            #use the earlier program 
            if save == True:
                    results.to_hdf('lap_results.h5',"lap_order_%s_dataset_%s_dataframe_%i_filtered"%(dataset,DF_name,dropped))
                    print('Laplacian Database created')
                    print("settings: Dataset = %s, Dataframe = %s, filtered by EIF = %i "%(dataset,DF_name,dropped))
            break
    return results

     #%% 
    
#Plotting Laplacian scores
def plot_lap(eif_df, remake = False, log=True):
    """
    Plots the results of laplace_order.
    Saves the plots
    Runs laplace_order if so required
    """
    #A bit of prepwork
    sns.set(style="whitegrid")
    #get the name for saving
    try:
        DF_name = get_df_name(eif_df)
    except (IndexError):
            DF_name = eif_df.nameDF_name = get_df_name(eif_df)     #get the name for saving
    
    #pull up the laplace orders
    df= laplace_order(eif_df, remake = remake)

    #Set up the variables
    feature = df.feature
    lapscore = df.laplacian_score

    #    Setting up the size of the plot, dependant on the number of outputs 
    fig, ax = plt.subplots(figsize=(10, 18))
    
    

    #Do the actual plotting
    sns.barplot(x = lapscore , y = feature,palette="icefire")
    #show_values_on_bars(ax, h_v="h")   #see values on bars. 
    sns.despine(left=True, bottom=False)   #removes spines
    if log == True:
        plt.xscale('log')
        plt.title("Laplacian scores of %s, sorted by rank, in log scale"%(DF_name),size='14')
        plt.savefig("pics/lap/Lapscore_%s_%i_filtered_logscale"%(DF_name,dropped),box_inches="tight")
    else:
        plt.title("Laplacian scores of %s, sorted by rank"%(DF_name),size='14')
        plt.savefig("pics/lap/Lapscore_%s_%i_filtered"%(DF_name,dropped),bbox_inches="tight")
    plt.show()
    
def show_values_on_bars(axs, h_v="v", space=0.4):
    """assistance if you want to show the values on bars. """
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = float(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif h_v == "h":
            for p in ax.patches:
                _x = 0.5
                #_x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = float(p.get_width())
                ax.text(_x, _y,  "%f.0 " %(value), ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
     #%% 
  
#weighted and normalised Calinski-Harabasz (comparison reasons)
def WNCH(X, cluster_predict, n_clusters,  L_r,):
    """
    Weighted Normalised Calinski-Harabasz Index
    input: 
    X = pandas dataframe, shape: (n_samples , n_features). Each is a single data point
    Will assume lap_part if none is given
    lables = an array, shaped to (n_samples), predicting the label for each sample
    Will assume cluster_predict if none is given
    
    Returns:
    score as a float
    possible adjustment proposed:First number explodes so start n_features at 0 rather then 1.
    This means the graph always starts at 0
    """
    
    n_features = X.shape[1]
    n_samples = X.shape[0]    #the sample size, or n
    
    extra_disp = 0.
    intra_disp = 0.
    
    mean = np.mean(X, axis=0).values # mean of the whole partial matrix, per feature


    for k in range(n_clusters):
        cluster_k = X[cluster_predict == k].values   # a matrix with just objects belonging to this cluster
        mean_k = np.mean(cluster_k, axis=0)                 # the mean vector for every feature
        extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)  #add to the trace of S_B (non diagonal cancel out)
        intra_disp += np.sum((cluster_k - mean_k) ** 2)              #add to the trace of S_W (non diagonal cancel out)
    
    y =  (extra_disp * (n_samples - n_clusters) * n_features) / (intra_disp * (n_clusters - 1) * L_r )
    #print('y =',y)
    if intra_disp == 0.:
        return 1
    else:
        return y
     #%%   
def WNCH2(X, cluster_predict, n_clusters,  L_r,):
    """
    Weighted Normalised Calinski-Harabasz Index, alternative
    input: 
    X = pandas dataframe, shape: (n_samples , n_features). Each is a single data point
    Will assume lap_part if none is given
    lables = an array, shaped to (n_samples), predicting the label for each sample
    Will assume cluster_predict if none is given
    
    Returns:
    score as a float
    ===============
    Adjustment made: First number explodes so start n_features at 0 rather then 1.
    This means the graph always starts at 0
    """
    
    n_features = X.shape[1]
    n_samples = X.shape[0]    #the sample size, or n
    
    extra_disp = 0.
    intra_disp = 0.
    
    mean = np.mean(X, axis=0).values # mean of the whole partial matrix, per feature


    for k in range(n_clusters):
        cluster_k = X[cluster_predict == k].values   # a matrix with just objects belonging to this cluster
        mean_k = np.mean(cluster_k, axis=0)                 # the mean vector for every feature
        extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)  #add to the trace of S_B (non diagonal cancel out)
        intra_disp += np.sum((cluster_k - mean_k) ** 2)              #add to the trace of S_W (non diagonal cancel out)
    
    #print('y =',y)
    if intra_disp == 0.:
        return 1
    else:
        y =  (extra_disp * (n_samples - n_clusters) * (n_features-1)) / (intra_disp * (n_clusters - 1) * L_r )
        return y
                
 #%%   
#non weighted nor normalised Calinski-Harabasz (comparison reasons)
def CH(X, cluster_predict, n_clusters):
    """
    Calinski-Harabasz Index
    input: 
    X = pandas dataframe, shape: (n_samples , n_features). Each is a single data point
    Will assume lap_part if none is given
    lables = an array, shaped to (n_samples), predicting the label for each sample
    Will assume cluster_predict if none is given
    
    Returns:
    score as a float
    """
    n_samples = X.shape[0]    
    extra_disp, intra_disp = 0., 0.
    mean = np.mean(X, axis=0).values # mean of the whole partial matrix, per feature


    for k in range(n_clusters):
        cluster_k = X[cluster_predict == k].values  # a matrix with just objects belonging to this cluster
        mean_k = np.mean(cluster_k, axis=0)                 # the mean vector for every feature
        extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
        intra_disp += np.sum((cluster_k - mean_k) ** 2)

    return (1. if intra_disp == 0. else
        extra_disp * (n_samples - n_clusters) /
        (intra_disp * (n_clusters - 1.)))
#%%


def LS_WNCH_SR(eif_df, k = 3, remake = False):
    """
    Laplacian Score-WNCH-Simple Ranking 
    ======================================
    Input:
    DF_name: Name of the dataframe, one of four possible right now, this is an F * N matrix
    F = number of features, N = number of datapoints)
    dropped: amount of data to be filtered with EIF
    k: the value of k for the k-means clustering test
    remake: Wether or not to remake the laplacian order matrix (memory intensive)
    
    ======================================
    Output: 
    
    ===============================================================
    Description:
    First hybrid method
    Builds n feature subsets (as many as featur)
    Uses KMeans
    =================================================================
    
    
    Based on Solario-Fernandez et al, 2019 [1], 
    Sources: 
    sklear: https://scikit-learn.org/stable/modules/clustering.html#calinski-harabasz-index
    Python notebook: https://nbviewer.jupyter.org/github/rasbt/pattern_classification/blob/
    master/dimensionality_reduction/projection/linear_discriminant_analysis.ipynb#LDA-in-5-steps
    Books
    """
    
    y_best = -np.inf
    y_list = []

    #laplace order
    lap_matrix = laplace_order(eif_df, remake = remake)
    max_features = lap_matrix.shape[0]

    for n_features in range(0,max_features):      #number of features we want to analyse over, start at 0 so add +1 where required

        
        #set some variables for WNCH 
        
        L_r = lap_matrix.iloc[n_features,1]             # Laplacian score associated with last r-th feature added or eliminated
        names = lap_matrix.iloc[:n_features+1].feature.values   # names of the top n features as an array 
        lap_part = eif_df[names]     #make a new (partial) dataframe containing only these features (This is S_0)!


        # Run a clustering algorhitm (Kmeans chosen here)
        scaler = StandardScaler()
        scaled = pd.DataFrame(scaler.fit_transform(lap_part),index=lap_part.index,columns=lap_part.columns)
        kmeans = KMeans(n_clusters= k).fit(scaled)  # set up a KMeans object, with expected amount of clusters, and fit to the partal dataframe
        cluster_predict = kmeans.predict(scaled)              #execute kmeans to predict, each object will be labeled with cluster number
        # cluster_centers = kmeans.cluster_centers_               #find the cluster centers (unused)
        
        # Calculate WNCH score:
        y = WNCH(scaled, cluster_predict, k, L_r)
        #   W2 = CH(lap_part, cluster_predict)  Obselete, for comparison sake
        #   CH_list.append(W2)
        if y > y_best:
            y_best = y
            S_best = names
        y_list.append(y)
        print('feature number: %i, y= %.2f'%(n_features+1,y))
    
    print(cluster_predict)
    data = {'column_name':S_best, 'y_score':y_list[:len(S_best)]}
    short = pd.DataFrame(data = data, dtype='float32')  
    data = {'column_name':names, 'y_score':y_list}
    long = pd.DataFrame(data = data)  
    return short, long
 
def SR_loader(eif_df, k, remake = False):
    """
    Input: 
        eif_df: EIF filtered dataframe
        k: number of clusters
        remake: remake the database entry or not?
    
    Output: 
        short and long dataframe belonging to LS_WNCH_SR
    
    Use: stores the results of LS_WNCH_SR,
    so that it does not need to be rerun every time
    """
    try:
        DF_name = eif_df.name
    except:
        DF_name = "custom"
    
    while True:
        try:
            if remake == True:
                print("New SR file requested")
                raise NameError('Remake')
            long = pd.read_hdf('hybrid_results.h5',"SR_results_%s_k_%i_long_filtered_%i"%(DF_name, k,dropped))
            short =  pd.read_hdf('hybrid_results.h5',"SR_results_%s_k_%i_short_filtered_%i"%(DF_name, k,dropped))
            print("succes, Hybrid results found")
            print("Settings: Database: %s, k = %i"%(DF_name, k))
            break
        except (KeyError,FileNotFoundError, NameError):
            if DF_name == "custom":
                short, long = LS_WNCH_SR(eif_df, k = k,remake=remake)
                break
            print("Failed to find Hybrid results, or remake requested")
            print("Settings: Database: %s, k = %i,"%(DF_name, k))
            short, long = LS_WNCH_SR(eif_df, k = k,remake=remake)            
            long.to_hdf('hybrid_results.h5',"SR_results_%s_k_%i_long_filtered_%i"%(DF_name, k,dropped))
            short.to_hdf('hybrid_results.h5',"SR_results_%s_k_%i_short_filtered_%i"%(DF_name, k,dropped))
            break
      
    return short, long
#%%
def plot_SR(eif_df, k, remake = False):
    """
    
    Parameters
    ----------
    eif_df : pandas dataframe 
        EIF filtered dataframe
    k : integer, optional
        number of clusters. The default is 3.
    remake : boolean, optional
         Remake the results or try to load previous results. The default is False.

    Returns
    -------
    short : pandas DataFrame
        Results of SR hybrid algorhitm
    long : pandas Dataframe
        SR hybrid results for all features

    """
    try:
        DF_name = eif_df.name
    except:
        DF_name = "custom"

    short, long = SR_loader(eif_df, k, remake = remake)
    
    fig, ax = plt.subplots(figsize=(8,8))
    sns.lineplot(data = (long['y_score']))
    plt.savefig("pics/SR/yplot_SR_%s_%i"%(DF_name,k),bbox_inches="tight")
    plt.show()
      
    fig, ax = plt.subplots(figsize=(8,8))
    #plt.xscale('log')
    plt.xlabel("WNCH score")
    plt.title("Most important %s features of SR hybrid, k = %i"%(DF_name,k),size='14')
    sns.barplot(x='y_score',y='column_name',data=short,palette='winter')
    plt.savefig("pics/SR/LS_WNCH_SR_%s_%i"%(DF_name,k),bbox_inches="tight")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(8,24))
    #plt.xscale('log')
    plt.xlabel("WNCH score")
    plt.title("%s features of SR hybrid, k = %i"%(DF_name,k),size='14')
    sns.barplot(x='y_score',y='column_name',data=long,palette='winter')
    plt.savefig("pics/SR/LS_WNCH_SR_complete_%s_%i"%(DF_name,k),bbox_inches="tight")
    plt.show()
    return short, long

#%%  
#Backward Elimination
y_list = []
rank_list = []
def LS_WNCH_BE(eif_df, k = 3, p = 30):
    
    """
    Laplacian Score-WNCH-Backward Elimination
    ======================================
    Input:
    DF_name: Name of the dataframe, one of four possible right now, this is an F * N matrix
    F = number of features, N = number of datapoints)
    dropped: amount of data to be filtered with EIF
    k: the value of k for the k-means clustering test
    remake: Wether or not to remake the laplacian order matrix (memory intensive)
    
    ======================================
    Output: 
    
    ===============================================================
    Description:
    First hybrid method
    Builds n feature subsets (as many as featur)
    
    =================================================================
    
    
    Based on Solario-Fernandez et al, 2019 [1], 
    Sources: 
    sklear: https://scikit-learn.org/stable/modules/clustering.html#calinski-harabasz-index
    Python notebook: https://nbviewer.jupyter.org/github/rasbt/pattern_classification/
    blob/master/dimensionality_reduction/projection/linear_discriminant_analysis.ipynb#LDA-in-5-steps
    Books
    """
    global y_list, rank_list, S_best  #Due to recursion, we will otherwise lose this information
    # global y_list, rank_list, S_best  #Due to recursion, we will otherwise lose this information
    y_list2 = []     #Alternative for plotting
    S_best = 0
   # y_list.append(np.nan)
    X_S0 = eif_df  #need to define a global name for the matrix we are using
    #   DF_name = get_df_name(X_S0)   #gets the name of the original matrix    
    n_clusters = k


    # What to return if there is only one feature?
    if X_S0.shape[1] == 1:
        print("only one feature")
        S_best = X_S0.columns[0]
        # Calculate Y_best score here aswell (for comparison with different k)
        lap_part = pd.DataFrame(X_S0[S_best])
        scaler = StandardScaler()
        scaled = pd.DataFrame(scaler.fit_transform(lap_part),index=lap_part.index,columns=lap_part.columns)
        kmeans = KMeans(n_clusters= n_clusters).fit(scaled)
        cluster_predict = kmeans.predict(scaled)
        # Find y score from here
        L_r = laplace_order(spec_eif).iloc[0,1]     # Assumption made for unmodified score (with one feature, score is relative)
        y_score = WNCH(scaled, cluster_predict, n_clusters, L_r)      #determine initial rank 
        
        data = {'column_name':S_best, 'y_score':y_score}
        S_best = pd.DataFrame(data = data, index = [0])  
        print("S_best =", S_best)
        return S_best, y_list

        
    else:
        lap_matrix = laplace_order(X_S0, remake = True,save= False)
        n_features = lap_matrix.shape[0]
        
        
        print(n_features," remaining features")
        flag = False

        #first cluster run
        names = lap_matrix.iloc[:n_features+1].feature.values   # names of the top n features as an array  (This is for S_0)
        lap_part = X_S0[names]     #make a new dataframe where features are ordered by laplace score (This is ind_rank)!
        initial = lap_part 
        
        # clustering algorhitm (Kmeans chosen here)
        scaler = StandardScaler()
        scaled = pd.DataFrame(scaler.fit_transform(lap_part),index=
                              lap_part.index,columns=lap_part.columns)
        kmeans = KMeans(n_clusters= n_clusters).fit(scaled)  # set up a KMeans object, with expected amount of clusters, and fit to the partal dataframe
        cluster_predict = kmeans.predict(scaled)              #execute kmeans to predict, each object will be labeled with cluster number
        
        #WNCH check to find y_best        
        L_r = lap_matrix.iloc[n_features-1,1]            # Laplacian score associated with last r-th feature added or eliminated
        y_best = WNCH(scaled, cluster_predict, n_clusters, L_r)      #determine initial rank 
        print("initial y_best: %.2f, %s"%(y_best,names[-1]))
        
        y_list.append(y_best)
        y_list2.append(y_best)
        
        counter = 0        
        n_features = n_features - 1   #remove the i'th feature
               

        for rank_nr in np.arange(0,n_features)[::-1]:    # Start at 
            L_r = lap_matrix.iloc[rank_nr,1]             # Laplacian score associated with last r-th feature added or eliminated
            names = lap_matrix.iloc[:rank_nr+1].feature.values   # plus one due to how the slices work (does not include the end point)
            lap_part = X_S0[names]    # S_o <-- indRank

            #run a clustering algorhitm over X_S_0
            # set up a KMeans object, with expected amount of clusters, and fit to the partal dataframe
            scaler = StandardScaler()
            scaled = pd.DataFrame(scaler.fit_transform(lap_part),index=
                                  lap_part.index,columns=lap_part.columns)
            kmeans = KMeans(n_clusters= n_clusters).fit(scaled) 
            cluster_predict = kmeans.predict(scaled)             
            y = WNCH(scaled, cluster_predict, n_clusters, L_r)

            if y > y_best:
                y_best = y
                print("Laplace rank: %i, new y_best: %.2f, %s"%((rank_nr+1),y_best,names[-1]))
                S_best = names  #Best = names of all the remaining items
                flag = True
            else:
                print('Laplace rank: %i, %s'%((rank_nr+1),names[-1])) 
            
            #some feedback mechanisms
            y_list.append(y)   #bonus for plotting
            y_list2.append(y)
            
            counter = counter +1
            if counter >= p:  #check for number of runs
                print('break, p is exceeded')
                break
        
        if flag == True:
            print('recursion')
            return LS_WNCH_BE(X_S0[S_best], k = k, p = p)
           # run alghoritm with X_s_best         
        
        elif S_best == 0:
            print('No improvement found within p = %i'%(p))
            names = initial.columns[::-1].values[:len(y_list2)]
            print(len(names),len(y_list2))
            data = {'column_name':names, 'y_score':y_list2}
            S_best = pd.DataFrame(data = data)
            S_best = S_best[::-1]
            print("S_best =", S_best)
            return S_best, y_list
            
            
        else:
            #No S_best found within p, 
            print('else')
            data = {'column_name':S_best, 'y_score':y_list2[::-1]}
            S_best = pd.DataFrame(data = data, dtype='float32')  
            print("S_best =", S_best)
            return S_best, y_list
    #End of loop!

#%%
            
    
def BE_loader(eif_df, k, p, remake=False):
    """" remake required for full y_list, not just of last recursion"""
    global y_list
    try:
        DF_name = eif_df.name
    except:
        DF_name = "custom"

    while True:
        try:
            if remake == True:
                print("New file requested")
                raise NameError('Remake')
            S_best =  pd.read_hdf('hybrid_results.h5',"BE_results_%s_k_%i_p_%i_filtered_%i"%(DF_name, k,p,dropped))
            y_list = np.load("ylist/%s%i%i%i.npy"%(DF_name, k,p,dropped)) #This makes it into a np list, even if it fails to load
            print("succes, Hybrid results found")
            print("Settings: Database: %s, k = %i, p = %i"%(DF_name, k,p))
            break
            
        except (KeyError,FileNotFoundError, NameError):
            if DF_name == "custom":
                S_best, y_list = LS_WNCH_BE(eif_df, k = k, p = p)
                break
            y_list = []  #make it an ordinary list
            print("Failed to find Hybrid results, or remake requested")
            print("Settings: Database: %s, k = %i, p = %i"%(DF_name, k,p))
            S_best, y_list = LS_WNCH_BE(eif_df, k = k, p = p)  #Apply LS_WNCH_BE
            np.save("ylist/%s%i%i%i.npy"%(DF_name, k,p,dropped),y_list)
            S_best.to_hdf('hybrid_results.h5',"BE_results_%s_k_%i_p_%i_filtered_%i"%(DF_name, k,p,dropped))
        break
      
    return S_best, y_list

#%%
def plot_BE(eif_df, k = 3, p = 30, remake=False):
    """
    

    Parameters
    ----------
    eif_df : pandas dataframe 
        EIF filtered dataframe
    k : integer, optional
        number of clusters. The default is 3.
    p : integer, optional
        maximum number of runs. The default is 30.
    remake : boolean, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    S_best : pandas DataFrame
        Shows the most important features, and relevant y scores
        note: first is always 1 at the moment
    y_list : TYPE
        List of how y develops. Simply for 

    """
    #Set lists to 0
    global y_list, rank_list
    y_list = []
    rank_list = []
    try:
        DF_name = eif_df.name
    except:
        DF_name = "custom"
    
    S_best, y_list = BE_loader(eif_df, k, p, remake=remake)
    
    fig = plt.subplots(figsize= (6,6))
    plt.plot(y_list)
    plt.show()

      
    fig, ax = plt.subplots(figsize=(8,8))
    #plt.xscale('log')
    plt.xlabel("WNCH score")
    plt.title("Most important %s features of BE hybrid, k = %i, p = %i"%(DF_name,k,p),size='14')
    sns.barplot(x='y_score',y='column_name',data=S_best,palette="mako")
    plt.savefig("pics/BE/LS_WNCH_BE_%s_%i_%i"%(DF_name,k,p),bbox_inches="tight")
    plt.show()
    
    return S_best, y_list
#%%

# Principle Feature Analysis part:

class PFA2(object):   #Initiate the object
    """
    Improved implementation of PFA
    Initial implementation had some mistakes and errors, such as:
    No use of correlation, which is required when variance between features can change a lot
    Naive implementation of PCA, selecting ALL features rather then 
    selectiung a number of components such that the amount of variance that needs to be explained
    is greater then say 90% (step 3)
    Doing K_means on the full dimensionality, as PCA didnt remove any

    """            
    def __init__(self, p_dif = 0 , cols=None, pov = 90):  #q is not yet filled in
        self.p_dif = p_dif 
        self.cols = cols                          #q = q
        self.pov = pov         # proportion of variance, percentage of the variance to conserve in PCA


    def fit(self, X):
        if not self.cols:                   #if q is not yet set
            self.cols = X.shape[1]          # q is the number of columns
        

        #New approach. Looking at the original paper, using a StandardScaler might be a bad idea. 
        # Hence, use covariance matrix or correlation matrix
        # correlation matrix is preferred in cases where the features have very different variances from each other, 
        # and where the regular covariance form will cause the PCA to put heavy weights on features with highest variances
        # https://plotly.com/python/v3/ipython-notebooks/principal-component-analysis/
    
        #standard scaler
        sc = StandardScaler()   #standard scaler program
        X_std = sc.fit_transform(X)  #fit the data, then transform it
    
        # step 1: covariance or correlation matrix
        cor_mat1 = np.corrcoef(X_std.T)        #  Same as X = X.corr() with pandas

        # step 2  compute Principle components and eigenvalues
        # 2a: determine eigen values, eigen vectors
        eig_vals, eig_vecs = np.linalg.eig(cor_mat1) #eigen values, eigen vectors

        # 2b: Make a list of (eigenvalue, eigenvector) tuples
        col_names = X.columns
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

        # 2c: Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_vals_sort = np.sort(eig_vals)[::-1]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        #eig_pairs.sort(reverse = True)

        # step 3: Determine retained variability and create A_q
        tot = sum(eig_vals)
        var_exp = [(i / tot)*100 for i in eig_vals_sort]
        cum_var_exp = np.cumsum(var_exp)
        self.var_exp = var_exp                            #addition for plotting!
        self.cum_var_exp = cum_var_exp                    #addition for plotting
        
        keep_q = cum_var_exp[cum_var_exp <= self.pov].shape[0] +1  #number of points required to get above pov value 
        
         # create A_q from eigen vectors
        A_q = eig_pairs[0][1].reshape(self.cols,1)  
        for i in np.arange(1,keep_q):
            A_q = np.hstack((A_q, eig_pairs[i][1].reshape(self.cols,1)))

        # Kmeans clustering of A_q
        kmeans = KMeans(n_clusters=keep_q + self.p_dif).fit(A_q)
        clusters = kmeans.predict(A_q)
        cluster_centers = kmeans.cluster_centers_  # mean of the cluster

        # for each cluster, find the vector closest to the mean of the cluster
        indices = np.array(())
        for center in cluster_centers:
            distance_list = np.array(())
            for vec in A_q:    
                dist = euclidean_distances(vec.reshape(-1, 1) , center.reshape(-1, 1))[0][0]
                distance_list = np.append(distance_list,dist)
            dl = distance_list
            #print(dl[dl == dl.min()])
            indices = np.append(indices, np.where( dl == dl.min()))

        indices = indices.astype('int')
        indices = np.unique(indices)  #sometimes multiple vectors can be closest. Hence uniques
        #col_names[[indices]].shape
        columns = col_names[indices]

        self.indices_ = indices
        self.columns_ = columns
        self.dataframe_ = X[self.columns_]

#%%
def pfa2_results(eif_df, run_nr = 15, p_dif = 0, pov = 90, remake = False):
    
    indice_array = np.zeros((run_nr, eif_df.shape[1])) #initiate an array
    indice_array[:] = np.nan
    for i in range(run_nr):
        pfa = PFA2(p_dif = p_dif, pov = pov)
        pfa.fit(eif_df)
        index = pfa.indices_
        indice_array[i,:len(index)] = index
   # print(indice_array)
    
    
    #Extract amount each row occurs, step by step for clarity:
    pandas = pd.DataFrame(indice_array)  #convert into pandas
    #Make a grid counting however often certain values happen
    result = pandas.apply(pd.value_counts).fillna(0)  #Applies a count, fills the numbers with no counts with zero
    summed = result.sum(axis=1)                       #Sum over an axis
    output = summed.sort_values(ascending = False)   #sort these values

    index = output.index.astype('int32')
    names = eif_df.columns[index]
    occurence = output.values.astype('int32')
    data = {'column_name':names, 'occurence':occurence}
    results_df = pd.DataFrame(data = data, dtype='float32')
    
    with plt.style.context('seaborn-whitegrid'):   #plot this once
        plt.figure(figsize=(6, 4))

        plt.hlines(pfa.pov, 0,pfa.cols, label = 'Variance of %i'%(pfa.pov))
        plt.bar(range(pfa.cols), pfa.var_exp, alpha=0.5, align='center',\
                label='individual explained variance',color='red')
        plt.step(range(pfa.cols), pfa.cum_var_exp, where='mid', 
                 label='cumulative explained variance',alpha=0.7)

        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.xlim(-0.3, pfa.cols+1)
        plt.ylim(0,102)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()

    return results_df
    
#output = pfa2_results(combi_eif)
#%%
# Code to find the results

def pfa2_loader(eif_df, run_nr = 15, p_dif = 0, pov = 90, remake = False):
    """
    Checks to see if a PFA result has been stored previously. If not, or remake = True, it will 
    make a PFA and return that.
    """
    try:
        DF_name = eif_df.name
    except:
        DF_name = "custom"
    
    while True:
        try:
            if remake == True:
                print("New file requested")
                raise NameError('Remake')
            results =  pd.read_hdf('PFA2_results.h5',"pfa2_results_%s_%i_runs_%i_%i"%(DF_name,run_nr,p_dif, pov))
            print("succes, PFA2 results found")
            print("settings: Dataframe = %s, run number = %i, p_difference = %i,pov = %i"%(DF_name, run_nr,p_dif,pov))
            break
        except (KeyError,FileNotFoundError, NameError):
            if DF_name == "custom":
                results = pfa2_results(eif_df, run_nr, p_dif, remake = remake)  
                break                
            print("Failed to find PFA2 results, creating database")
            results = pfa2_results(eif_df, run_nr, p_dif, remake = remake)               #use the earlier program 
            results.to_hdf('PFA2_results.h5',"pfa2_results_%s_%i_runs_%i_%i"%(DF_name,run_nr,p_dif,pov))
            print('PFA2 Database created')
            print("settings: Dataframe = %s, run number = %i, p_difference = %i,pov = %i"%(DF_name, run_nr,p_dif,pov))
            break
 
    return results
#%%

def plot_PFA(eif_df, run_nr = 15, p_dif = 0, pov = 90, remake = False):
    """
    Parameters
    ----------
    eif_df : pandas dataframe
        This needs to befiltered with anomaly detection such as EIF already
    run_nr : Integer, optional
        Amount of runs to do. The default is 15.
    p_dif : int, optional
        p > q, slightly higher number of features is in some cases needed. The default is 0.
    pov : int, optional
        Retained variability in %. The default is 90.
    remake : Boolean, optional
        Set to True if you want to remake previously stored data. The default is False.

    Returns
    -------
    dataframe with column numbers of eif_df and occurence of these columns over all runs combined
    a bar plot showing the occurence parameters over run_nr runs. This plot has been adjusted to 
    only display parameters which occur atleast 0.10 * run_nr amount of times (10%)

    """
    try:
        DF_name = eif_df.name
    except:
        DF_name = "custom"
    #    
        
    results_df = pfa2_loader(eif_df, run_nr = run_nr, p_dif = p_dif, pov = pov, remake = remake)   

    # -------------------------------------------------------------
    # We now have loaded (and if required created) a dataframe with results for n_list
    # If we want to change n_list, we must make sure it's saved and loaded by a new name, or replace the old
    sns.set(style="whitegrid")


    #Set up the variables
    results_top = results_df[results_df.iloc[:,1].values >= run_nr * 0.2]


    #    Setting up the size of the plot, dependant on the number of outputs 
    fig, ax = plt.subplots(figsize=(6,5))

    #Do the actual plotting
    sns.barplot(x='occurence',y='column_name',data=results_top,palette ="viridis_r")
    ax.set(xlim= [0,run_nr+1],xlabel="Occurence",ylabel = "Feature name")
    plt.title("Top %s features, %i runs, p_dif = %i, pov = %i" %(DF_name,run_nr,p_dif,pov),size='14')
    sns.despine(left=True, bottom=False)   #removes spines
    plt.savefig("pics/PFA/PFA2_%s_%i_%i_%i"%(DF_name,run_nr,p_dif,pov),bbox_inches="tight")
    plt.show()
    
    return results_df
#for i in dataframes:
 #   PFA2_plot(i, 100)
#plot_PFA(phot_eif, 100)


#%%



#%%
def cluster_BE(eif_df, k,p,a=0,b=1):

    BE_best = BE_loader(eif_df,k,p)
    names = BE_best.iloc[:,0].values
    BE_part = eif_df[names]
    
    kmeans = KMeans(n_clusters= k).fit(BE_part)  # set up a KMeans object, with expected amount of clusters, and fit to the partal dataframe
    cluster_predict = kmeans.predict(BE_part)          
    
    sns.set_style("whitegrid")
    cmap = 'prism'
    sns.scatterplot(x=BE_part.columns[a],y=BE_part.columns[b], hue = cluster_predict,data = BE_part,palette=cmap)
    plt.show()
    
def cluster_DF(eif_df, k):

    BE_best = BE_loader(eif_df,k)
    names = BE_best.iloc[:,0].values
    BE_part = eif_df[names]
    
    kmeans = KMeans(n_clusters= k).fit(BE_part)  # set up a KMeans object, with expected amount of clusters, and fit to the partal dataframe
    cluster_predict = kmeans.predict(BE_part)          
    
    sns.set_style("whitegrid")
    cmap = 'prism'
    
    sns.pairplot(inv_eif.sample(1000), corner = True,plot_kws=dict(s=10, edgecolor="None", linewidth=1), diag_kind="kde", diag_kws=dict(shade=True))
    plt.show()
#%%
    """
remake = True
klist = [2,4,6,8]
for i in dataframes:
    for k in klist:
        plot_SR(i, k, remake)
#%%
#BE_plot(eif_df, k, p, remake=False):
klist = [2,3,4]
for i in dataframes:
    for k in klist:
        plot_BE(i, k,25, remake)
#%%
flist = [3,6,9,12]
for i in dataframes:
    for k in flist:
        PFA_plot(i, k, run_nr = 250, remake = remake)
#%%
for i in dataframes:
    plot_lap(i, remake = remake, log=True)
    plot_lap(i, remake = remake, log=False)
#%%
#BE_plot(eif_df, k, p, remake=False):
klist = [2,3,4]
for i in dataframes:
    for k in klist:
        plot_BE(i, k,30, remake)
        """
        