
'''

Code for Week 02 prac

Adapted from Peter Norvig

Last modified by f.maire@qut.edu.au on 2022/01/31

'''

import sys
import itertools, time

#if sys.version < '3.5':
#    raise Exception("You should be using Python 3.5 or later")

def imright(h1, h2):
    "House h1 is immediately right of h2 if h1-h2 == 1."
    return h1-h2 == 1

def nextto(h1, h2):
    "Two houses are next to each other if they differ by 1."
    return abs(h2-h1) == 1

def zebra_puzzle_naive():
    '''
    Naive implementation of an exhaustive search to find a solution to the
    zebra puzzle. This version will take a long time to find a solution.
    I stopped the computation after 10 minutes on my laptop.
    
    Return a tuple (WATER, ZEBRA) indicating their house numbers.
    Return (None, None) is no solution was found.
    '''
    # Use some extra variables for readability
    houses = first, _, middle, _, _ = [1, 2, 3, 4, 5]
    orderings = list(itertools.permutations(houses)) 

    for red, green, ivory, yellow, blue in orderings:
        for Englishman, Spaniard, Ukranian, Japanese, Norwegian in orderings:
            for dog, snails, fox, horse, ZEBRA in orderings:
                for coffee, tea, milk, oj, WATER in orderings:
                    for OldGold, Kools, Chesterfields, LuckyStrike, Parliaments in orderings:
                        # test all the constraints
                        if ( Englishman == red # constraint 2
                             and Spaniard == dog # constraint 3
                             and coffee == green # constraint 4
                             and Ukranian == tea # constraint 5
                             and imright(green, ivory) # constraint 6
                             and OldGold == snails # constraint 7
                             and Kools == yellow # constraint 8
                             and milk == middle # constraint 9
                             and Norwegian == first # constraint 10
                             and nextto(Chesterfields, fox) # constraint 11
                             and nextto(Kools, horse) # constraint 12
                             and LuckyStrike == oj # constraint 13
                             and Japanese == Parliaments # constraint 14
                             and nextto(Norwegian, blue) # constraint 15
                             ):
                            return WATER, ZEBRA

def zebra_puzzle_cut1():
    '''
    Naive implementation of an exhaustive search to find a solution to the
    zebra puzzle. This version will take a long time to find a solution.
    I stopped the computation after 10 minutes on my laptop.
    
    Return a tuple (WATER, ZEBRA) indicating their house numbers.
    Return (None, None) is no solution was found.
    '''
    # Use some extra variables for readability
    houses = first, _, middle, _, _ = [1, 2, 3, 4, 5]
    orderings = list(itertools.permutations(houses)) 

    for red, green, ivory, yellow, blue in orderings:
        if not imright(green, ivory): # constraint 6
            continue
        for Englishman, Spaniard, Ukranian, Japanese, Norwegian in orderings:
            if (Englishman != red # constraint 2
                or Norwegian != first # constraint 10
                or not nextto(Norwegian, blue) # constraint 15
                ):
                continue
            for dog, snails, fox, horse, ZEBRA in orderings:
                if Spaniard != dog: # constraint 3
                    continue
                for coffee, tea, milk, oj, WATER in orderings:
                    if (coffee != green # constraint 4
                        or Ukranian != tea # constraint 5
                        or milk != middle # constraint 9
                        ): 
                        continue
                    for OldGold, Kools, Chesterfields, LuckyStrike, Parliaments in orderings:
                        # test all the constraints
                        if ( OldGold == snails # constraint 7
                             and Kools == yellow # constraint 8
                             and nextto(Chesterfields, fox) # constraint 11
                             and nextto(Kools, horse) # constraint 12
                             and LuckyStrike == oj # constraint 13
                             and Japanese == Parliaments # constraint 14
                             ):
                            return WATER, ZEBRA



def zebra_puzzle_gen():
    '''
        What is returned by this thing ?!
        
    '''
    houses = first, _, middle, _, _ = [1,2,3,4,5]
    orderings = list(itertools.permutations(houses)) #1
    genExp = ((WATER, ZEBRA)
            for (red, green, ivory, yellow, blue) in orderings
            for (Englishman, Spaniard, Ukranian, Japanese, Norwegian) in orderings
            for (dog, snails, fox, horse, ZEBRA) in orderings
            for (coffee, tea, milk, oj, WATER) in orderings
            for (OldGold, Kools, Chesterfields, LuckyStrike, Parliaments) in orderings
            if Englishman == red                    #2
            if Spaniard == dog                      #3
            if coffee == green                      #4
            if Ukranian == tea                      #5
            if imright(green, ivory)                #6
            if OldGold == snails                    #7
            if Kools == yellow                      #8
            if milk == middle                       #9
            if Norwegian == first                   #10
            if nextto(Chesterfields, fox)           #11
            if nextto(Kools, horse)                 #12
            if LuckyStrike == oj                    #13
            if Japanese == Parliaments              #14
            if nextto(Norwegian, blue)              #15
            )
    return genExp

def zebra_puzzle_gen_cut():
    '''
        What is returned by this thing ?!
        
    '''
    houses = first, _, middle, _, _ = [1,2,3,4,5]
    orderings = list(itertools.permutations(houses)) #1
    genExp = ((WATER, ZEBRA)
            for (red, green, ivory, yellow, blue) in orderings
            if imright(green, ivory)                #6
            for (Englishman, Spaniard, Ukranian, Japanese, Norwegian) in orderings
            if Englishman == red                    #2
            if Norwegian == first                   #10
            if nextto(Norwegian, blue)              #15
            for (dog, snails, fox, horse, ZEBRA) in orderings
            if Spaniard == dog                      #3
            for (coffee, tea, milk, oj, WATER) in orderings
            if coffee == green                      #4
            if Ukranian == tea                      #5
            if milk == middle                       #9
            for (OldGold, Kools, Chesterfields, LuckyStrike, Parliaments) in orderings
            if OldGold == snails                    #7
            if Kools == yellow                      #8
            if nextto(Chesterfields, fox)           #11
            if nextto(Kools, horse)                 #12
            if LuckyStrike == oj                    #13
            if Japanese == Parliaments              #14
            )
    return genExp
## - - - - - - - - - - - - - - - - - -

def ex_0():
    '''
    Test of the naive exhaustive search
    
    '''
    t0 = time.time()
    (w,z) = zebra_puzzle_naive()
    t1 = time.time()    
    print ('w, z = {},{}'.format(w,z))
    print ('Search took {} seconds'.format(t1-t0))

def ex_1():
    t0 = time.time()
    g=zebra_puzzle_gen_cut()
    (w,z) = next(g)
    t1 = time.time()    
    print ('w, z = {},{}'.format(w,z))
    print ('Search took {} seconds'.format(t1-t0))


if __name__ == "__main__":
    ex_0()
    ex_1()
    

