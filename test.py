def printxx(xx):
    for i in range(0,len(xx)):
        print(xx[i])

def printyy(yy):
    for i in range(0, len(yy)):
        print(yy[i])



xx = [[-5,-4,-3,-2,-1,0,1,2,3,4,5] for i in range(7)]
yy = [[-3,-2,-1,0,1,2,3] for i in range(11)]

printxx(xx)
printyy(yy)

c1 = [[0 for i in range(11)]for i in range(7)]
printxx(c1)


radius = 2

print("MAKING CIRCLE... \n \n")

for c in range(11):
    for r in range(7):
        temp = ((xx[r][c] - radius/2)**2) + (yy[c][r]**2)
        # print("r: ", r, " c: ", c ,"temp: ", temp)
        if temp < (radius**2):

            # print("less than")
            # print("r**2: ", (radius**2))
            c1[r][c] = 1
            # printxx(c1)
        else:
            c1[r][c] = 0

printxx(c1)


# c2 = [[0 for i in range(11)]for i in range(7)]
# print("c2")
# printxx(c2)
# print()

# for c in range(11):
#     for r in range(7):
#         temp = ((xx[r][c] + radius/2)**2) + (yy[c][r]**2)
#         # print("r: ", r, " c: ", c ,"temp: ", temp)
#         if temp < (radius**2):

#             # print("less than")
#             # print("r**2: ", (radius**2))
#             c2[r][c] = 1
#             # printxx(c2)
#         else:
#             c2[r][c] = 0

# printxx(c2)
