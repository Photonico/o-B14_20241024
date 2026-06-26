import numpy as numpy
import matplotlib.pyplot as plt
from py4vasp import Calculation

#Calculations
Calc = Calculation.from_path("./")

#Band data
Bands = Calc.band.to_dict(selection="KPOINTS_OPT")

#Extract labels and positions
labels = Bands.get('kpoint_labels')
position = Bands.get("kpoint_distances")
k_label = []
k_pos = []
cnt_label = ''
for i in range(len(labels)):
    if(i==0):
        k_label.append(labels[i])
        k_pos.append(position[i])
        cnt_label = labels[i]
    if(i>0 and labels[i]!='' and cnt_label!=labels[i]):
        k_label.append(labels[i])
        k_pos.append(position[i])
        cnt_label = labels[i]

def HOMO_LUMO(Bands):
    HOMO = -999
    LUMO = 999
    Count = 0
    HOMO_kpt = 0
    LUMO_kpt = 0
    for B in Bands:
        for i in range(len(B)):
            if(B[i]<=0 and B[i]>HOMO):
                HOMO = B[i]
                HOMO_kpt = Count
            if(B[i]>0 and B[i]<LUMO):
                LUMO = B[i]
                LUMO_kpt = Count
        Count += 1
    return LUMO-HOMO

BANDGAP = HOMO_LUMO(Bands.get("bands"))
print(f"Band gap is: {BANDGAP} eV")


#Plot the figure
plt.rcParams["font.family"] = "Liberation Serif"
fig, ax1 = plt.subplots(1,1,figsize=(8, 5))

ax1.plot(Bands.get("kpoint_distances"),Bands.get("bands"), color="orange")
ax1.axhline(y=0, linestyle="--", color = "k")
ax1.set_ylim(-4,4)
ax1.set_xlim(0,max(Bands.get("kpoint_distances")))
ax1.set_ylabel("Energy (eV)", fontsize=20)
#ax1.set_title("Charged", fontsize=20)
plt.setp(ax1, xticks=k_pos, xticklabels=k_label)
ax1.tick_params(axis='both', which='major', direction="in", right = True, length=7, labelsize=14)
for x in k_pos:
    ax1.axvline(x=x, linewidth=1, color ="k")

plt.tight_layout()
#plt.show()
plt.savefig("aspect_oB14_bilayer_bandstructure.pdf",dpi=600)