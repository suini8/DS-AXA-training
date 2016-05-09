# -*- coding: utf-8 -*-
"""
Created on Sun May 08 16:58:15 2016

@author: Alex
"""

#A. donuts
def donuts(count):
    # +++your code here+++
    if count < 10:
        result = "Number of donuts: " + str(count)
    else:
        result = "Number of donuts: many"
    return result
    

#B. both_ends

def both_ends(s):
    # +++your code here+++
    if len(s) > 2 :
        result = s[0]+s[1]+s[-2]+s[-1]
    else:
        result = ""
    return result
    


# C. fix_start
def fix_start(s):
    # +++your code here+++
    x = s[0]
    result = x + s[1:].replace(x, "*")

    return result



#D. MixUp
def mix_up(a, b):
    # +++your code here+++
    a_2 = b[:2]+a[2:]
    b_2 = a[:2]+b[2:]
    
    result = a_2+" "+b_2
    
    return result
    
# test functions
def test(got, expected):
    prefix = 'OK' if got == expected else ' X'
    # !r prints a Python representation of the strings (complete with quotes)
    print ' {} got: {!r} expected: {!r}'.format(prefix, got, expected)    
    


def main():
    print 'donuts'
    # Each line calls donuts, compares its result to the expected for that call.
    test(donuts(4), 'Number of donuts: 4')
    test(donuts(9), 'Number of donuts: 9')
    test(donuts(10), 'Number of donuts: many')
    test(donuts(99), 'Number of donuts: many')

    print
    print 'both_ends'
    test(both_ends('spring'), 'spng')
    test(both_ends('Hello'), 'Helo')
    test(both_ends('a'), '')
    test(both_ends('xyz'), 'xyyz')

  
    print
    print 'fix_start'
    test(fix_start('babble'), 'ba**le')
    test(fix_start('aardvark'), 'a*rdv*rk')
    test(fix_start('google'), 'goo*le')
    test(fix_start('donut'), 'donut')

    print
    print 'mix_up'
    test(mix_up('mix', 'pod'), 'pox mid')
    test(mix_up('dog', 'dinner'), 'dig donner')
    test(mix_up('gnash', 'sport'), 'spash gnort')
    test(mix_up('pezzy', 'firm'), 'fizzy perm')

main()  




#D. verbing
def verbing(s):
    # +++your code here+++
    if len(s)>3:
        if s[-3:]=="ing":
            s = s+"ly"
        else:
            s = s+"ing"
    return s


#E. not_bad
def not_bad(s):
    # +++your code here+++
    if (s.find("not") < s.find("bad")) & (s.find("not") != -1) & (s.find("bad") != -1):
        result = s[:s.find("not")]+"good"+s[s.find("bad")+3:]
    else:
        result=s
    return result



#F. front_back
def front_back(a, b):
    # +++your code here+++
    def funct(word):
        length = len(word)/2 + len(word)%2
        part1 = word[:length]
        part2 = word[length:]
        return part1,part2
    
    word_a = funct(a)
    word_b = funct(b)
    result = word_a[0] + word_b[0] + word_a[1] + word_b[1]

    return result
    




def main():
    print 'verbing'
    test(verbing('hail'), 'hailing')
    test(verbing('swiming'), 'swimingly')
    test(verbing('do'), 'do')
    
    print
    print 'not_bad'
    test(not_bad('This movie is not so bad'), 'This movie is good')
    test(not_bad('This dinner is not that bad!'), 'This dinner is good!')
    test(not_bad('This tea is not hot'), 'This tea is not hot')
    test(not_bad("It's bad yet not"), "It's bad yet not")

    print
    print 'front_back'
    test(front_back('abcd', 'xy'), 'abxcdy')
    test(front_back('abcde', 'xyz'), 'abcxydez')
    test(front_back('Kitten', 'Donut'), 'KitDontenut')

  
main()  
