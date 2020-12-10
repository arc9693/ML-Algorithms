import numpy as np
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def find_entropy(df):
    Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = df[Class].value_counts()[value]/len(df[Class])
        entropy += -fraction*np.log2(fraction)
    return entropy
  
  
def find_entropy_attribute(df,attribute):
  Class = df.keys()[-1]   #To make the code generic, changing target variable class name
  target_variables = df[Class].unique()  #This gives all 'Yes' and 'No'
  variables = df[attribute].unique()    #This gives different features in that attribute (like 'Hot','Cold' in Temperature)
  entropy2 = 0
  print("\t",end='')
  print("\t",end='')
  for target_variable in target_variables:
      print('\t',target_variable,end='')
  print('\t','Total',end='')
  print('\t','Entropy fraction',end='\n')
  for variable in variables:
      print('\t\t',variable,end='')
      entropy = 0
      for target_variable in target_variables:
          num = len(df[attribute][df[attribute]==variable][df[Class] ==target_variable])
          den = len(df[attribute][df[attribute]==variable])
          print('\t',num,end='')
          fraction = num/(den+eps)
          entropy += -fraction*log(fraction+eps)
      print('\t',den,end='')
      fraction2 = den/len(df)
      entropy2 += -fraction2*entropy
      if(entropy<0):
          print('\t',0.0,end='\n')
      else:
          print('\t',entropy,end='\n')
  return abs(entropy2)


def find_winner(df):
    Entropy_att = []
    IG = []
    total_entropy = find_entropy(df)
    print("Total entropy:\t",total_entropy)
    for key in df.keys()[:-1]:
        print(bcolors.UNDERLINE + "Attribute: "+key+bcolors.ENDC)
        attribute_entropy = find_entropy_attribute(df,key)
        Entropy_att.append(attribute_entropy)
        IG.append(total_entropy-attribute_entropy)
        print("\t\t","Gain("+key+"):",total_entropy-attribute_entropy)
    print("\nMax Gain: ",df.keys()[:-1][np.argmax(IG)])
    return df.keys()[:-1][np.argmax(IG)]
  
  
def get_subtable(df, node, value):
  return df[df[node] == value]

discreteTargetFunction=""
def buildTree(df,tree=None): 
    # Class = df.keys()[-1]   #To make the code generic, changing target variable class name
    
    #Here we build our decision tree

    #Get attribute with maximum information gain
    print("CURRENT DATAFRAME\n-------------------")
    print(df,'\n')
    print("CALCULATIONS FOR SELECTING NODE")
    print("-------------------------------")
    node = find_winner(df)
    print("-----------------------------\n")

    print(bcolors.WARNING + "NODE SELECTED: "+ node + bcolors.ENDC+"\n")

    #Get distinct value of that attribute e.g Salary is node and Low,Med and High are values
    attValue = np.unique(df[node])
    
    #Create an empty dictionary to create tree    
    if tree is None:                    
        tree={}
        tree[node] = {}
    
   #We make loop to construct a tree by calling this function recursively. 
    #In this we check if the subset is pure and stops if it is pure. 

    for value in attValue:
        
        subtable = get_subtable(df,node,value)
        print(bcolors.OKGREEN + 'BRANCH: '+ value + bcolors.ENDC)
        clValue,counts = np.unique(subtable[discreteTargetFunction],return_counts=True)                        
        
        if len(counts)==1: #Checking purity of subset
            tree[node][value] = clValue[0]
            print("CURRENT DATAFRAME\n-------------------")
            print(subtable,'\n')
            print("The subset is Pure.")
            print(bcolors.WARNING + "NODE SELECTED: "+ clValue[0] + bcolors.ENDC+"\n")                                           
        else:        
            tree[node][value] = buildTree(subtable) #Calling the function recursively 
                   
    return tree
  
  
if __name__ == "__main__":
    filename = input('File name: ')
    discreteTargetFunction = input('Target variable: ')
    hasIndex = input('Has index?(Y/N) : ')
    df = pd.read_csv(filename)
    if(hasIndex=='Y'):
        indexName = input('Index name: ') 
        df = df.set_index(indexName)
    print(bcolors.OKGREEN + 'ROOT' + bcolors.ENDC)
    buildTree(df)