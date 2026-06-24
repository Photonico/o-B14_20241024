import matplotlib.pyplot as plt

FILE = open("./OSZICAR","r")
NULL = 0
TIME = []
ENERGY = []
TEMP = []
while(NULL<50):
    LINE = FILE.readline()
    if("T=" in LINE):
        SLINE = LINE.split()
        TIME.append(int(SLINE[0]))
        ENERGY.append(float(SLINE[6]))
        TEMP.append(float(SLINE[2]))
    if(LINE.strip("\n")==""):
        NULL +=1

fig, ax = plt.subplots(2,1)
ax[0].plot(TIME,ENERGY)
#ax[0].set_xlabel("Time (fs)")
ax[0].set_ylabel("Energy (eV)")
ax[0].set_xlim(0,5500)
ax[0].set_xticks([])


ax[1].plot(TIME,TEMP, color="r")
ax[1].set_xlabel("Time (fs)")
ax[1].set_ylabel("Temperature (K)")
ax[1].set_xlim(0,5500)

plt.tight_layout()
plt.savefig("./AIMD.png", dpi=600)


