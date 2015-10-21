
house = read.table("../ex1/housing.data")
x = house[,1:13]
y = house[,14]

x = cbind(1,x)
