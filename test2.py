#this is just another test to make sure git is working on my desktop
#Plot Creation

x1 = [0]*(len(Q))
y1 = [0]*(len(Q))

for i in range(0, len(x1)):
    x1[i] = 5*i
    Qdose = 0
    for j in range(0, len(Q[0])):
        if (Q[i][j] > Qdose):
            Qdose = Q[i][j]
            y1[i] = j
print('x1', x1)
print('y1', y1)
plt.plot(x1, y1)

plt.xlabel('Blood Glucose Level')
plt.ylabel('Insulin Dosage')

plt.title('Q-Learning Dosage Recomendations')

plt.show()