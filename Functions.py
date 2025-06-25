import pickle
from ClassesML2 import *


def level_loader(level_loc):
    with open(level_loc, 'rb') as f: 
        level = pickle.load(f)
    return level
    

def car_from_parameters(parameters):
    # Car(5,40,20,([100,200,255]),1000,10,level.location,3)
    return Car(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7])






