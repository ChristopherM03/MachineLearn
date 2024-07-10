row0 = [' ', 'x', 'o']
row1 = [' ', 'o', ' ']
row2 = ['x', ' ', 'x']
board = [row0, row1, row2]
for row in board:
    for cell in row:
       print(cell + ' ', end="")
    #print

print('-------')
for row in board:
    for cell in row:
        print('|', end="")
        print(cell, end="")
    print('|')
    print('-------')
