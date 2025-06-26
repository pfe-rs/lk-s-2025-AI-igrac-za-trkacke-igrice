from pathlib import Path
import pickle
from ClassesML2 import *


def level_loader(level_loc: Path | str):
    with open(level_loc, 'rb') as f: 
        level = pickle.load(f)
    return level
    

def car_from_parameters(parameters):
    # Car(5,40,20,([100,200,255]),1000,10,level.location,3)
    return Car(parameters[0],parameters[1],parameters[2],parameters[3],parameters[4],parameters[5],parameters[6],parameters[7])


def save_record(array, filename):
    """
    Save array clipped by 50 elements at the start and 50 at the end if longer than 100.
    If array has 100 or fewer elements, save the entire array.
    """
    if len(array) > 160:
        to_save = array[80:-80]  # Remove 50 from start, 50 from end
    else:
        to_save = array  # Save all if <= 100

    with open(filename, "wb") as f:
        pickle.dump(to_save, f)
    
    print(f"Saved {len(to_save)} elements to {filename}")


def load_record(filename):
    """Load array from pickle file."""
    with open(filename, "rb") as f:
        return pickle.load(f)
