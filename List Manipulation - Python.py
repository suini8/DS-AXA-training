# -*- coding: utf-8 -*-
"""
Created on Sun May 08 16:16:49 2016

@author: Alex
"""

#A. match_ends
def match_ends(words):
    # +++your code here+++
    number = len(words)
    count_a = 0
    for i in range(number):
        if len(words[i]) >= 2 and  words[i][0] == words[i][-1] :
                count_a = count_a +1

    return count_a


# B. front_x
def front_x(words):
    # +++your code here+++
    number = len(words)
    list_x = []
    list_abc = []
    for i in range(number):
        if words[i][0] == "x" :
        
            list_x.insert(len(list_x),words[i])
        else:
            list_abc.insert(len(list_abc),words[i])

    list_x.sort() 
    list_abc.sort() 
    list_x.extend(list_abc)
    return list_x 
 

# C. sort_last
def sort_last(tuples):
    # +++your code here+++
    dict_tuples = {}
    for i in range(len(tuples)):
        #dict_tuples[i] = tuples[i][-1]
        dict_tuples[tuples[i][-1]] = i
    
    order = []   
    for i in range(len(tuples)):    
        order.append(dict_tuples[sorted(dict_tuples.keys())[i]])
    
    list_tuples = []
    for i in range(len(tuples)):    
        list_tuples.insert(i,tuples[order[i]])  
    return list_tuples

result=sort_last([(1, 7), (1, 3), (3, 4, 5), (2, 2)])
print(result) 


# Test functions 
def test(got, expected):
    prefix = 'OK' if got == expected else ' X'
    # !r prints a Python representation of the strings (complete with quotes)
    print ' {} got: {!r} expected: {!r}'.format(prefix, got, expected)

def main():
    print 'match_ends'
    test(match_ends(['aba', 'xyz', 'aa', 'x', 'bbb']), 3)
    test(match_ends(['', 'x', 'xy', 'xyx', 'xx']), 2)
    test(match_ends(['aaa', 'be', 'abc', 'hello']), 1)

    print
    print 'front_x'
    test(front_x(['bbb', 'ccc', 'axx', 'xzz', 'xaa']),
        ['xaa', 'xzz', 'axx', 'bbb', 'ccc'])
    test(front_x(['ccc', 'bbb', 'aaa', 'xcc', 'xaa']),
        ['xaa', 'xcc', 'aaa', 'bbb', 'ccc'])
    test(front_x(['mix', 'xyz', 'apple', 'xanadu', 'aardvark']),
        ['xanadu', 'xyz', 'aardvark', 'apple', 'mix'])
        
        
main()





#D. remove_adjacent
def remove_adjacent(nums):
    # +++your code here+++
    list_unique = []
    for i in range(len(nums)):
        if nums[i] not in list_unique:
            list_unique.append(nums[i])
    return list_unique


#E. linear_merge
def linear_merge(list1, list2):
    # +++your code here+++
    list_order = []
    n1 = 0
    n2 = 0

    while ((n1 < len(list1)) & (n2 < len(list2))):
        if list1[n1] < list2[n2]:
           list_order.insert(len(list_order),list1[n1])
           n1 = n1 + 1
        else:
           list_order.insert(len(list_order),list2[n2])
           n2 = n2 + 1
                
    if n1 > len(list1)-1 :
        while n2 < len(list2):
            list_order.insert(len(list_order),list2[n2])
            n2 = n2 + 1
    else:
        while n1 < len(list1):
            list_order.insert(len(list_order),list1[n1])
            n1 = n1 + 1  
    return list_order


# Test functions
def main():
    print 'remove_adjacent'
    test(remove_adjacent([1, 2, 2, 3]), [1, 2, 3])
    test(remove_adjacent([2, 2, 3, 3, 3]), [2, 3])
    test(remove_adjacent([]), [])

    print
    print 'linear_merge'
    test(linear_merge(['aa', 'xx', 'zz'], ['bb', 'cc']),
        ['aa', 'bb', 'cc', 'xx', 'zz'])
    test(linear_merge(['aa', 'xx'], ['bb', 'cc', 'zz']),
        ['aa', 'bb', 'cc', 'xx', 'zz'])
    test(linear_merge(['aa', 'aa'], ['aa', 'bb', 'bb']),
        ['aa', 'aa', 'aa', 'bb', 'bb'])



main()


#Appendix:
""" misunderstanding of the first exercise:
the function hereunder is counting the number of letter in a word of a list 
only if the word has more than 2 letters and if the first and last letter are the same.
"""
def match_ends2(words):
    # +++your code here+++
    number = len(words)
    results = []
    for word_x in range(number):
        if len(words[word_x]) > 2 :
            if words[word_x][0] == words[word_x][-1]:
                results.insert(word_x,len(words[word_x]))
            else:
                results.insert(word_x,"err_2")
        else:
            results.insert(word_x,"err_1")
    return results
    
test(match_ends2(['', 'x', 'xyz', 'xyx', 'xx']), ['err_1', 'err_1', 'err_2', 3, 'err_1'])

