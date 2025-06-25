import pickle
from ClassesML2 import *


def level_loader(level_loc):
    with open(level_loc, 'rb') as f: 
        level = pickle.load(f)
    return level
    

def car_from_parameters(parameters: dict):
    # Car(5,40,20,([100,200,255]),1000,10,level.location,3)
    return Car(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7])


def save_record(array, filename):
    """
    Save all elements except the last 100 to a pickle file, if array has more than 100 elements.
    If array has 100 or fewer elements, save the entire array.
    """
    if len(array) > 100:
        to_save = array[:-100]
    else:
        to_save = array  # Save all if <= 100

    with open(filename, "wb") as f:
        pickle.dump(to_save, f)
    
    print(f"Saved {len(to_save)} elements to {filename}")


def load_record(filename):
    """Load array from pickle file."""
    with open(filename, "rb") as f:
        return pickle.load(f)
