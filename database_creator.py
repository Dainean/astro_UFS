# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:56:53 2020

@author: Dainean
"""

#Prepare the python system 
import pandas as pd  
import numpy as np 
import fnmatch   #For filtering
import os   #move around in our OS
from astropy.io import fits  #Working with fits
from astropy.cosmology import WMAP9 as cosmo  #Cosmology calculators
import itertools as it #iteration / combination trick used
import seaborn as sb
import matplotlib.pyplot as plt

#Working directory control
cwd = os.getcwd() 
#Working directory control
cwd = os.getcwd() 
print("Initial working directory is:", cwd) 
if '/Users/users/verdult/Thesis/thesis' in cwd:
    print("Working at kapteyn, changing to data directory")
    os.chdir('/net/virgo01/data/users/verdult/Thesis')  #This is for kapteyn
if 'data' in cwd:
    print("Working in kapteyn data folder")
if 'Dropbox' in cwd:
    print("Working at home, changing to onedrive folder")
    os.chdir('D:\Onedrive\Thesis') 
if 'Dainean' in cwd:
    print("Working at home, changing to onedrive folder")
    os.chdir('D:\Onedrive\Thesis') 
if 'Onedrive' in cwd:
    print("Working in onedrive folder")
cwd = os.getcwd() 

print("Current working directory is:", cwd) 

#%%
def pandafy3(filename):  #load up the whole fit file as a pandafile
        
    filename = filename  #which file? #222,617 KB
     
    while True:
        try:
            if remake == True:
                print("New file requested")
                raise NameError('remake')
            dfm = pd.read_hdf('support/SupportDB.h5', 'initial_db')  #Read the initial dataframe
            print("file found")
            break
        except (FileNotFoundError,KeyError,NameError):
            print("creating new file")
                    
            simple = fits.open(filename) #open it
            data = simple[1].data  #data bit
            hdr = simple[1].header #header bit
            cols = simple[1].columns  #The columns from .fits file as an object
            simple.close()
        
            coln = cols.names   #Names of the columns
            colnpd = pd.Series(coln)  #Convert to a pandas series (so we can search the strings)
            
            columns = colnpd
            
        
            B = np.zeros(len(data))  #Initiate an array of the correct size
            for i in columns:
                C = data.field(i)   #Read the data from a specific coloum
                B = np.column_stack((B,C))
            D = np.delete(B,0,1)  #We did the first column twice
            
            # create the dataframe
            df = pd.DataFrame(D, index = data.field(0), columns = columns.values)
            df.to_hdf('support/tester.h5', 'test_db')  # 195,492 KB
            break
    
    # create the dataframe
    df = pd.DataFrame(D, index = data.field(0), columns = columns.values)
    return(df)
#%%
#This cell will contain the main variables and functions

filename_1 = 'fits/combined/DS-Sersic-SA-kCorr_m4.fits'  #which file? #222,617 KB
#we are at version 4 right now. 

simple = fits.open(filename_1) #open it
data = simple[1].data  #data bit
hdr = simple[1].header #header bit
cols = simple[1].columns  #The columns from .fits file as an object

coln = cols.names   #Names of the columns
colnpd = pd.Series(coln)  #Convert to a pandas series (so we can search the strings)
simple.close()

def pandafy(data,columns):  #Colns must be all the columns you want to include
    ARG = columns.index      #!!!!  Does this do anything?
    B = np.zeros(len(data))  #Initiate an array of the correct size
    for i in columns:
        C = data.field(i)   #Read the data from a specific coloum
        B = np.column_stack((B,C))
    D = np.delete(B,0,1)  #We did the first column twice
    df = pd.DataFrame(D, index = data.field(4), columns = columns.values, dtype='float32')
    return(df)

def pandafy2(filename):  #load up the whole fit file as a pandafile
        
    filename = filename  #which file? #222,617 KB
     
    simple = fits.open(filename) #open it
    data = simple[1].data  #data bit
    hdr = simple[1].header #header bit
    cols = simple[1].columns  #The columns from .fits file as an object
    simple.close()

    coln = cols.names   #Names of the columns
    colnpd = pd.Series(coln)  #Convert to a pandas series (so we can search the strings)
    
    columns = colnpd
    

    B = np.zeros(len(data))  #Initiate an array of the correct size
    for i in columns:
        C = data.field(i)   #Read the data from a specific coloum
        B = np.column_stack((B,C))
    D = np.delete(B,0,1)  #We did the first column twice
    
    # create the dataframe
    df = pd.DataFrame(D, index = data.field(0), columns = columns.values)
    return(df)

def fittify(df,filename='ThesisDB_selected.fits'):  #say which dataframe you want to turn into a fit file
    holder = []
    for i in range(df.columns.values.size):
        holder.append(fits.Column(name=df.columns.values[i], format='D', array=df.iloc[:,i]))

    cols = fits.ColDefs(holder)
    hdu = fits.BinTableHDU.from_columns(cols)

    hdu.writeto(filename,overwrite=True)
#%%
#Check for initial dataframe 
remake = False   #remake the dataframe even if it exists?
#remake = True
while True:
    try:
        if remake == True:
            print("New file requested")
            raise Exception()
        dfm = pd.read_hdf('support/InitialDB.h5', 'initial_db')  #Read the initial dataframe
        print("file found")
        break
    except (FileNotFoundError,KeyError,NameError):
        print("creating new file")
        dfm = pandafy(data,colnpd[2:430])  #Turning the whole dataset into a pandas dataframe, keep out any strings
        dfm.to_hdf('support/InitialDB.h5', 'initial_db')  # 195,492 KB
        fittify(dfm, "thesis_gama.fits")
        break

def pandafy3(filename, remake = False):  #load up the whole fit file as a pandafile
    
    while True:
        try:
            if remake == True:
                print("New file requested")
                raise NameError('remake')
            dfm = pd.read_hdf('support/SupportDB.h5', 'initial_db')  #Read the initial dataframe
            print("file found")
            break
        except (FileNotFoundError,KeyError,NameError):
            print("creating new file")
            dfm = pandafy2])  #Turning the whole dataset into a pandas dataframe, keep out any strings
            dfm.to_hdf('support/InitialDB.h5', 'initial_db')  # 195,492 KB
            break
     
    simple = fits.open(filename) #open it
    data = simple[1].data  #data bit
    hdr = simple[1].header #header bit
    print(hdr)
    cols = simple[1].columns  #The columns from .fits file as an object
    simple.close()      # close the file again

    coln = cols.names   #Names of the columns
    colnpd = pd.Series(coln)  #Convert to a pandas series (so we can search the strings)
    
    columns = colnpd
    

    B = np.zeros(len(data))  #Initiate an array of the correct size
    for i in columns:
        C = data.field(i)   #Read the data from a specific coloum
        B = np.column_stack((B,C))
    D = np.delete(B,0,1)  #We did the first column twice
    
    # create the dataframe
    df = pd.DataFrame(D, index = data.field(0), columns = columns.values, dtype='float32')
    return(df)

# Extinction dataframe
remake = False
#remake = True
    
while True:
    try:
        if remake == True:
            print("New file requested")
            raise Exception()
        extinc = pd.read_hdf('support/SupportDB.h5', 'extinction')
        print("file found")
        break
    except:
        print("creating new file")
        extinc = pandafy2('fits/GalacticExtinction.fits')
        extinc.to_hdf('support/SupportDB.h5', 'extinction')
        break


# SDSS Dataframe
remake = False
#remake = True
    
while True:
    try:
        if remake == True:
            print("New file requested")
            raise Exception()
        SDSS = pd.read_hdf('support/SupportDB.h5', 'SersicSDSS')
        print("file found")
        break
    except:
        print("creating new file")
        SDSS = pandafy2('fits/SersicCatSDSS.fits')
        SDSS.to_hdf('support/SupportDB.h5', 'SersicSDSS')
        break

# UKID dataframe
remake = False
while True:
    try:
        if remake == True:
            print("New file requested")
            raise Exception()
        UKID = pd.read_hdf('support/SupportDB.h5', 'SersicUKIDSS')
        print("file found")
        break
    except:
        print("creating new file")
        UKID = pandafy2('fits/SersicCatUKIDSS.fits')
        UKID.to_hdf('support/SupportDB.h5', 'SersicUKIDSS')
        break
#%%
#This cell will contain the main variables and functions

filename_1 = 'fits/combined/DS-Sersic-SA-kCorr_m4.fits'  #which file? #222,617 KB
#we are at version 4 right now. 

simple = fits.open(filename_1) #open it
data = simple[1].data  #data bit
hdr = simple[1].header #header bit
cols = simple[1].columns  #The columns from .fits file as an object

coln = cols.names   #Names of the columns
colnpd = pd.Series(coln)  #Convert to a pandas series (so we can search the strings)
simple.close()

def pandafy(data,columns):  #Colns must be all the columns you want to include
    ARG = columns.index      #!!!!  Does this do anything?
    B = np.zeros(len(data))  #Initiate an array of the correct size
    for i in columns:
        C = data.field(i)   #Read the data from a specific coloum
        B = np.column_stack((B,C))
    D = np.delete(B,0,1)  #We did the first column twice
    df = pd.DataFrame(D, index = data.field(4), columns = columns.values, dtype='float32')
    return(df)

def pandafy2(filename):  #load up the whole fit file as a pandafile
        
    filename = filename  #which file? #222,617 KB
     
    simple = fits.open(filename) #open it
    data = simple[1].data  #data bit
    hdr = simple[1].header #header bit
    cols = simple[1].columns  #The columns from .fits file as an object
    simple.close()

    coln = cols.names   #Names of the columns
    colnpd = pd.Series(coln)  #Convert to a pandas series (so we can search the strings)
    
    columns = colnpd
    

    B = np.zeros(len(data))  #Initiate an array of the correct size
    for i in columns:
        C = data.field(i)   #Read the data from a specific coloum
        B = np.column_stack((B,C))
    D = np.delete(B,0,1)  #We did the first column twice
    
    # create the dataframe
    df = pd.DataFrame(D, index = data.field(0), columns = columns.values, dtype='float32')
    return(df)

def fittify(df,filename='ThesisDB_selected.fits'):  #say which dataframe you want to turn into a fit file
    holder = []
    for i in range(df.columns.values.size):
        holder.append(fits.Column(name=df.columns.values[i], format='D', array=df.iloc[:,i]))

    cols = fits.ColDefs(holder)
    hdu = fits.BinTableHDU.from_columns(cols)

    hdu.writeto(filename,overwrite=True)

#%%
#Updated database creation!

dfm = pd.read_hdf('support/InitialDB.h5', 'initial_db')  #Read the initial dataframe

#-------------------------------------------------------------
GALMAG = dfm[dfm.columns[dfm.columns.str.contains("GALMAG_")]] #Grab all the Magnitudes
GALMAG = GALMAG[GALMAG > -9999]   #create a new dataframe, where all the "bad" values are replaced by NaN
dis = cosmo.comoving_distance(dfm['Z'])    #Comoving distances, using cosmo package and applied to Z

dfm2 = GALMAG  #needless renaming, but hassle to rewrite
#Starting out with 6 columns, 
#iloc[:,0:6]

dfm2['CATAID'] = dfm['CATAID']
dfm2['RA'] = dfm['RA']
dfm2['DEC'] = dfm['DEC']

dfm2['NQ'] = dfm['NQ']   #Add Redshift quality to the new dataframe (can remove later)
dfm2['Z'] = dfm['Z']   #Add Redshift Z to the new dataframe
dfm2['Distance (Mpc)'] = dis   #Add the distances we found to the dataframe.

dfm3 = dfm2.iloc[:,9:]  #yet another new dataframe, keeping only the latter from dfm2 
#(easier this way then to create a new df)
#-------------------------------------------------------------
#prepare and filter out some bad values: 
galr = dfm[dfm.columns[dfm.columns.str.contains("GALR90_")]] #Grab all the 90% radia
galr = galr[galr > -9999]  #filter out invalid values

galRE = dfm[dfm.columns[dfm.columns.str.contains("GALRE_")]] #Grab all the Effective Radia in arcsec
galRE = galRE[galRE > -9999]  #filter out invalid values

galmu = dfm[dfm.columns[dfm.columns.str.startswith("GALMU")]] #Gather all the surface brightnesses 
galmu = galmu[galmu > -9999]

bands = "ugrizYJHK"  #All the bands we will iterate over
arcsec = (2*np.pi)/(360*3600)  #one arcsec in radians
#kpc2 = (((np.sin(arcsec)*dis)**2)*10**6).value  #arcsec^2 converted to kpc^2 for each distance  

#-------------------------------------------------------------
#various band information, 10 bits of data over 9 bands = 90 columns 
#iloc[:,6:87]

minradius = 0 # minimum radius in kpc, can't have negative
maxradius = 10**18 #max radius in kpc, set extremely high as we switched to outlier detection instead
j = 0   #alternative counter for iterations

for i in bands: #Add to dfm3, 
    #dfm3['RELMAG_%s'%(i)] = dfm['GALMAG_%s'%(i)]  #relative magnitude, can drop this
    
    #Absolute magnitude, based on distance, kcorrection and galactic foreground extinction
    dfm3['ABSMAG%s'%(i)] = 5 + (GALMAG['GALMAG_%s'%(i)] -5*np.log10((dis.value*10**6))) \
    - dfm['KCORR_%s'%(i)] - extinc.loc[:,'A_u':'A_K_UKIDSS'].iloc[:,j] 
    
    #Radius (kpc) that contains 90% of light from galaxy
    r = (np.sin(galr['GALR90_%s'%(i)]*arcsec)*dis.value)*10**3         
    dfm3['size90%s'%(i)] = r[((r > minradius) & (r < maxradius))]     #Radius (kpc) that fits 90% of the light)
    
     #Radius (kpc) that contains 50% of the light of the galaxy
    r = (np.sin(galRE['GALRE_%s'%(i)]*arcsec)*dis.value)*10**3         #Filter out unrealistic values
    dfm3['sizeRE%s'%(i)] = r[((r > minradius) & (r < (maxradius/4)))]   #Radius (kpc) where light is at 50%
    
    #sersic index, no adjustments
    dfm3['SersicIndex%s'%(i)] = dfm['GALINDEX_%s'%(i)]                #good as is
   
    # ================================================================
    #dfm3['SersicIndexErr%s'%(i)] = dfm['GALINDEXERR_%s'%(i)]          #Error on sersic index, added 14-02-2019
    
    #This is added with the idea that the error on the sersic index will also say something about irregularities in the shape. 
    #Is this correct? Not correct according to article. Other errors more important, need to filter those out
     # ================================================================
    
    #ellipticity,  no adjustments
    dfm3['Ellipticity%s'%(i)] = dfm['GALELLIP_%s'%(i)]                #good as is
    
    #Absolute magnitute at 10 Re   #per band: Mv = mv - 2.5*log10((distance / 10 pc)**2) - kcorr
    dfm3['ABSMAG10RE%s'%(i)] = (dfm['GALMAG10RE_%s'%(i)] +5 -5*np.log10((dis.value*10**6))) - dfm['KCORR_%s'%(i)] 
    
    #Central surface brightness in (absmag / arcsec^2)  #No sense changing this  
    dfm3['MU@0%s'%(i)] = dfm['GALMU0_%s'%(i)]  
    
    #Effective surface brightness at effective radius (absmag / arcsec^2) #No sense changing this
    dfm3['MU@E%s'%(i)] = dfm['GALMUE_%s'%(i)] 
    
    #Average Effective surface brightness within effective radius (absmag / arcsec^2)
    dfm3['MUEAVG%s'%(i)] = dfm['GALMUEAVG_%s'%(i)]  
    
    j += 1



#-------------------------------------------------------------
#Convert the colours and add them to the dataframe, 36 in total
#[:,87:123]

b=np.arange(len(bands))                                  #to make an combinations series
combi = pd.Series(list(it.combinations(b,2)))   #praise to atomh33ls at stackoverflow
for i in combi:                                
    dfm3['%s-%s'%(bands[i[0]],bands[i[1]])] = (dfm3['ABSMAG%s'%(bands[i[0]])]-dfm3['ABSMAG%s'%(bands[i[1]])])
#-------------------------------------------------------------
#Exrtract some flux info some line fluxes
equivW = dfm[dfm.columns[dfm.columns.str.endswith("EW")]] #Grab all the continua

#-------------------------------------------------------------
#Add spectral information, 52 columns added
#[:,123:175]

#add the 4000 A break strength 
dfm3['D4000N'] = dfm['D4000N']  

#add all the equivalent widhts (Which is measured flux / background radiation)
for i in range(len(equivW.columns)):
    dfm3[equivW.columns[i]] = equivW.iloc[:,i]
    
# ==================================
# Database has been constructed. Now to drop any NaN values

df3 = dfm3.dropna()  #Drop any rows that have NaN in them. This brings us down to 42289 rows. 

#small change to before. 36769 becomes 34981
df = df3[df3 > -99999].dropna()  #some Equivalent widths have dummy values, here we drop them  36769 rows

# Save partial forms
phot = df.iloc[:,6:87] 
colour = df.iloc[:,87:123]
spectral = df.iloc[:,123:175] 