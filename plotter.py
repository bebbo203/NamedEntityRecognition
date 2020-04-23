import matplotlib.pyplot as plt
import sys

f = open(sys.argv[1])

train_loss = []
val_loss = []
microprec = []
macroprec = []
recall = []
f1 = []
epoch = []

for i, riga in enumerate(f.readlines()):
    t = riga.split(",")
    train_loss.append(float(t[0]))
    val_loss.append(float(t[1]))
    microprec.append(float(t[2]))
    macroprec.append(float(t[3]))
    recall.append(float(t[4]))
    f1.append(float(t[5]))
    epoch.append(i)



fig, axs = plt.subplots(2)

for elem in zip(epoch, train_loss):
    print(str(elem[0]) + ", "+ str(elem[1]))



axs[0].plot(epoch, train_loss, label="train_loss")
axs[0].plot(epoch, val_loss, label="val_loss")

axs[0].legend()



axs[1].plot(epoch, microprec, label="microprecision")
axs[1].plot(epoch, macroprec, label="macroprecision")
axs[1].plot(epoch, recall, label="recall")
axs[1].plot(epoch, f1, label="f1-score")
axs[1].legend()
plt.show()
