import pygame
import sys
import numpy as np
import random
import time
import math
import pickle
import os
from ClassesML2 import *









def quad_list_from(input_tree,quads):
    output_list=[]
    for i in range(4):
        if(quads[i]):
            output_list.extend(input_tree[i])
    return output_list

def quad_tree_from_list(input_list, level):
    width, height = level.proportions
    boolean_lines = []

    for line in input_list:
        quads = [False, False, False, False]

        for x, y in [(line.x1, line.y1), (line.x2, line.y2)]:
            if x < width / 2:
                if y < height / 2:
                    quads[0] = True  # NW
                else:
                    quads[2] = True  # SW
            else:
                if y < height / 2:
                    quads[1] = True  # NE
                else:
                    quads[3] = True  # SE

        boolean_lines.append(quads)

    return boolean_lines

def get_chosen_ones(input_list,boolean_lines,decided_quad):
    chosen_list = []

    for j in range(len(boolean_lines)):
        for i in range(4):
            if boolean_lines[j][i] and decided_quad[i]:
                chosen_list.append(input_list[j])
                break  # Ne mora dalje proveravati, linija je već izabrana

    return chosen_list
        