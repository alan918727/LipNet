def get_list_safe(l, index, size):
    ret = l[index:index+size]

    # while the index+size is over the length of array, it back to the first of array and append.
    while size - len(ret) > 0:
        ret += l[0:size - len(ret)]
                 
    return ret
    
