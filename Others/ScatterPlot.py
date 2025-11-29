import matplotlib.pyplot as plt

x= [1,2,3,4,5,6,7,8,9,10]
y= [112,24,256,184,200, 50,20,240,360,156]

plt.scatter(x,y, color='red', marker='+')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Scatter Plot')
plt.show()