#!/usr/bin/env python

import numpy as np

class ArrayList(object):
    def __init__(self, content=[], unique=True):
        self.content = []
        self.shape = None
        self.unique = unique
        
        for array in content:
            self.append(T)
    
    def _convert_element(self, i):
        element = self.content[i]
        
        return np.array(element)
    
    def __getitem__(self, i):
        return self._convert_element(i)
    
    def __iter__(self):
        for i in range(len(self.content)):
            yield self._convert_element(i)
    
    def __contains__(self, value):
        return list(value) in self.content
    
    def __len__(self):
        return len(self.content)
    
    # T: numpy array
    def append(self, array):
        if self.shape is None:
            self.shape = array.shape
        
        assert self.shape == array.shape, "the shape of the array to be inserted {} is not the same as other arrays {}".format(array.shape, self.shape)
                
        arrayT = list(array)
        
        if self.unique:
            if arrayT not in self.content:
                self.content.append(arrayT)
        else:
            self.content.append(arrayT)
    
    #def get_list(self):
        #array = []
        #for i in range(len(self.content)):
            #array.append(self._convert_element[i])
        
        #return array
    
    def remove(self, value):
        value = list(value)
        self.content.remove(value)
        
class ArrayDict(object):
    def __init__(self, unique=True):
        self.content = {}
        self.unique = unique
    
    def __getitem__(self, key):
        return list(self.content[key])
    
    def __setitem__(self, key, value):
        key = set(key)
        self.content[key] = ArrayList(value, unique)
    
    def keys():
        return [np.array(i) for i in self.content.keys()]
    
    def has_key(self, key):
        key = set(key)
        return key in self.content.keys()