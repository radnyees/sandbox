import pandas as pd
import numpy as np
import json
import sys
np.warnings.filterwarnings('ignore') # ignore warnings

# instead of having to regenerate subject mapping, read from saved file
sub_map = json.load(open('files/subj_map.json','r'))

# define path to excel file and various variable names needed
#mathfiles = ['T1_Num_Run01', 'T1_Num_Run02', 'T2_Num_Run01', 'T2_Num_Run02']
#mathfiles = ['T1_Rhyming']#, 'T2_Rhyming']


# ask for user input instead, only works for one file at a time
# trial = input('Ses (T1, T2): ')
task = input('Task (SLD, SLI, SSD, SSI, VLD, VLI, VSD, VSI): ')
run = input('Run (01, 02): ')
adhdfiles = [task+'_Run'+run]


bidpath = input('\nPath to bids folder \n   ./ for current directory \n  ../ for back a directory: ')
for file in adhdfiles:
    file_ext = '.csv'
    #trial, task, run = file.split("_",2)
    ##trial, task = file.split("_",2)
    #if task == 'Rhyming':
    #    run=''
    #else:
    #    run=run[3:]
    print('\nCSV file: '+file+file_ext)
    stim = 'files/{0}-stims_Run{1}.csv'.format(task, run)
    print('Stim file: '+stim)
    
    # read in csv file
    df = pd.read_csv('files/'+file+file_ext)
    
    # rename columns for manageability
    df = df.rename(columns={"Dummy.OnsetTime": "dummy", "Letters1Back.OnsetTime": "1onset", "Letters2Back.OnsetTime": "2onset",
                       "NO1.OnsetTime": "no1onset", "NO2.OnsetTime": "no2onset", "Target1.OnsetTime": "base1onset", "Target2.OnsetTime": "base2onset",         
                         "Letters1Back.ACC": "1acc", "Letters2Back.ACC": "2acc", "Target1.ACC": "base1acc", "Target2.ACC": "base2acc",
                            "Letters1Back.RT": "1rt", "Letters2Back.RT": "2rt", "Target1.RT": "base1rt", "Target2.RT": "base2rt",})


    #subtract 600 ms from all baseline onsets
    df['baseonset'] = df['base1onset'].fillna(df['base2onset'])
    df['baseonset'] = df['baseonset'] - 600
    
    # merge appropriate columns and drop extra columns
    df['letonset'] = df['1onset'].fillna(df['2onset'])
    df['noonset'] = df['no1onset'].fillna(df['no2onset'])
    df['onset1'] = df['letonset'].fillna(df['baseonset'])
    df['onset'] = df['onset1'].fillna(df['noonset'])
    df['letacc'] = df['1acc'].fillna(df['2acc'])
    df['baseacc'] = df['base1acc'].fillna(df['base2acc'])
    df['accuracy1'] = df['letacc'].fillna(df['baseacc'])
    df['letrt'] = df['1rt'].fillna(df['2rt'])
    df['basert'] = df['base1rt'].fillna(df['base2rt'])
    df['nort'] = df['NO1.RT'].fillna(df['NO2.RT'])
    df['noacc'] = df['NO1.ACC'].fillna(df['NO2.ACC'])

    for x in range(len(df['nort'].values)):
            if  df['nort'].iloc[x] == 0:
                df['noacc'].iloc[x] = 1

    df['accuracy'] = df['accuracy1'].fillna(df['noacc'])
    df['response_time1'] = df['letrt'].fillna(df['basert'])
    df['response_time'] = df['response_time1'].fillna(df['nort'])

    df['onset'] = df['onset'] - df['dummy']
    df['onset'] = df['onset'] - 12000
    
    df = df[['Subject', 'onset', 'accuracy', 'response_time']]
    
    # convert times from ms to s
    df[['onset', 'response_time']] = df[['onset', 'response_time']].applymap(lambda x: x / 1000)
    df
    
    # break up tables for each subject
    subjects = df['Subject'].unique()
    
    subjects.sort() # might not need this anymore since we read in sub nums from a dict
    print("Total subjects: " + str(len(subjects)))
    
    for subj in subjects: 
        df0 = df[(df['Subject'] == subj)].reset_index(drop=True)
    
        # error checking
        for x in range(len(df0['response_time'].values)):
            if  df0['response_time'].iloc[x] == 0:
                df0['response_time'].iloc[x] = 'n/a'
    
        # drop extra columns, rearrange and add in prime and target stim columns
        #df0 = df0.drop(columns=['Subject', 'cresp', 'resp', 'ft'])
        df0 = df0[['onset', 'accuracy', 'response_time']]
        stims = pd.read_csv(stim)
        df0 = pd.concat([df0, stims],axis=1)

        df0 = df0[['onset', 'duration', 'trial_type', 'correct_response', 'accuracy', 'response_time', 'stimulus', 'position']]
    
        # replace NaN values with n/a
        for col in list(df0):
            df0[col] = df0[col].replace(np.nan, 'n/a')
        
        # reference sub_map dict first to figure out correct number
        sub_num = sub_map[str(subj)]
            
        # get new subject number
        if sub_num < 10:
            new_subj = '0' + str(sub_num)
        else:
            new_subj = str(sub_num)
    
        # save to tsv file
        fname = bidpath+'bids/sub-{}/ses-T1/func/sub-{}_ses-T1_task-{}_events.tsv'.format(new_subj, new_subj, task)
        # use --pretend flag to see what files will be made
        # need to implement better flag checking
        if len(sys.argv)>1 and sys.argv[1] == '--pretend':
            print(fname)
        else:
            df0.to_csv(fname, sep='\t', index=False)
    
