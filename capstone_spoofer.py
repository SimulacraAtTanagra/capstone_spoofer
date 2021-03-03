# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 18:00:58 2020

@author: shane
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm 

def random_var(size, p=None, var=None):
    """Generate n-length ndarray of genders."""
    if not p:
        # default probabilities
        p = (0.5,0.5)
    if not var:
        var = ("M", "F")
    return np.random.choice(var, size=size, p=p)
def get_rand(listc,df,year,p=None):
    if not p:
        p=[df[year].iloc[fielddict[i]] for i in listc]
    size=int(df[year].iloc[0])
    return(random_var(size,p,listc))
def count_enr(var1,var2,df,year):
    x=int(df[year].iloc[fielddict[var1]])/int(df[year].iloc[fielddict[var2]])
    return([x,1-x])
def enr_perc(perc,var,df,year):
    return(int(int(df[year].iloc[fielddict[var]])*df[year].iloc[fielddict[perc]]))
def give_perc(val,df,year):
    a=df[year].iloc[fielddict[val]]
    b=1-a
    return([a,b])
def gen_data(ylist,df):
    ylist2=[]
    for i in ylist:
        yr=i
        applieds=get_rand(applied,df,int(i),p=applied_perc)
        accepteds=get_rand(accepted,df,int(i),p=count_enr("Acceptances","Applications",df,int(i)))
        enrolleds=get_rand(enrolled,df,int(i),p=count_enr("Enrollments","Acceptances",df,int(i)))
        genders=get_rand(gender,df,int(i),p=give_perc('Gender -Men',df,int(i)))
        ethnicities=get_rand(ethnicity,df,int(i))        
        highschools=get_rand(hs,df,(int(i)))
        choice=get_rand(college_choice,df,(int(i)))
        majors=get_rand(major,df,(int(i)))
        sats=get_rand(sat_score,df,(int(i)))
        pells=get_rand(pell,df,int(i),p=give_perc('Percent Receiving Pell Grants',df,int(i)))
        averages = np.random.normal(loc=df[int(i)].iloc[35], scale=0.025, size=int(df[int(i)].iloc[0]))
        xdf=pd.DataFrame({'year':yr,'applied':applieds,'accepted':accepteds,'enrolled':enrolleds,
                          'gender':genders,'ethnicity':ethnicities,'HS':highschools,'Choice order':choice,
                          'Major School':majors,"SAT Band":sats,"Pell recipient":pells,"HS Avg":averages})
        ylist2.append(xdf)
    return(ylist2)
def spoof_main(dataloc):
    #Gathering aggregate data
    df = pd.read_excel(dataloc).fillna(0)
    
    #Using categories of aggregate data to assign variables to randomize
    year=['2014','2015','2016','2017','2018','2019']
    applied=accepted=enrolled=yielded=pell=[True,False]
    applied_perc=[1,0]
    admissions_pipe=''
    gender=[i for i in df.Year.values if 'Gender' in i]
    ethnicity=[i for i in df.Year.values if 'Ethnicity' in i]
    hs=[i for i in df.Year.values if 'Highschool' in i]
    college_choice=[i for i in df.Year.values if 'College Choice' in i]
    major=[i for i in df.Year.values if 'Major School' in i]
    average=''
    sat_score=[i for i in df.Year.values if 'SAT Scores' in i]
    fielddict={i:ix for ix,i in enumerate(df.Year.values)}
    
    # loading the training dataset  
    dfx=pd.concat(gen_data(year,df))
    
    #binning data but copying dataframe first
    dfy=dfx[dfx.year!=999999]
    # 1 for true, 2 for false for bool values, numbers of objects in dict order for all others
    yeardict={int(i):ix for ix,i in enumerate(year)}
    yeardict.update({str(i):ix for ix,i in enumerate(year)})
    dfy.year=dfy.year.apply(lambda x: yeardict[x])
    dfy.applied=dfy.applied.apply(lambda x: int(x))
    dfy.accepted=dfy.accepted.apply(lambda x: int(x))
    dfy.enrolled=dfy.enrolled.apply(lambda x: int(x))
    ethnicitydict={i:ix for ix,i in enumerate(ethnicity)}
    hsdict={i:ix for ix,i in enumerate(hs)}
    college_choicedict={i:ix for ix,i in enumerate(college_choice)}
    majordict={i:ix for ix,i in enumerate(major)}
    sat_scoredict={i:ix for ix,i in enumerate(sat_score)}
    genderdict={i:ix for ix,i in enumerate(gender)}
    dfy.gender=dfy.gender.apply(lambda x: genderdict[x])
    dfy.ethnicity=dfy.ethnicity.apply(lambda x: ethnicitydict[x])
    dfy.HS=dfy.HS.apply(lambda x: hsdict[x])
    dfy['Choice order']=dfy['Choice order'].apply(lambda x: college_choicedict[x])
    dfy['Major School']=dfy['Major School'].apply(lambda x: majordict[x])
    dfy['SAT Band']=dfy['SAT Band'].apply(lambda x: sat_scoredict[x])
    dfy['Pell recipient']=dfy['Pell recipient'].apply(lambda x: int(x))
    
    # defining the dependent and independent variables 
    Xtrain = dfy[[ 'accepted', 'gender', 'ethnicity', 'HS',
           'Choice order', 'Major School', 'SAT Band', 'Pell recipient', 'HS Avg']] 
    ytrain = dfy[['enrolled']] 
    
    # building the model and fitting the data 
    log_reg = sm.Logit(ytrain, Xtrain).fit() 
    
    pred = np.array(log_reg.predict(Xtrain), dtype=float)
    table = np.histogram2d(np.array(ytrain.enrolled.to_list()), pred, bins=2)[0]
    accuracy=(table[0][0]+table[1][1])/(np.sum(table))
    sensitivity=table[0][0]/(table[0][0]+table[1][0])
    specificity=table[1][1]/(table[0][1]+table[1][1])
    return(table,accuracy,sensitivity,specificity)
