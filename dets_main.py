import func
import dets_func
import de_func
import numpy as np
import matplotlib.pyplot as plt

n = 10
d = 2
de_f = 0.9
dets_f = [0.5, 0.9, 1.5]
c = 0.9
iter = 1000
g = 100
ep = 0.5
ep = 1

# 関数と画像の名前を設定
function = func.ackley
grp_num = 2

de = de_func.DE(n, d, de_f, c, function, iter, g)
dets = dets_func.DETS(n, d, dets_f, c, function, iter, g, ep)

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["font.size"] = 15
plt.rc('legend', fontsize=13)

x = np.arange(g)
de_y = de.get_average_best_solution_list()
dets_y = dets.get_average_best_solution_list()
plt.plot(x, de_y, label="DE, F=" + str(de_f))

str_f = ""
for i in range(len(dets_f)):
    str_f = str_f + str(dets_f[i])
    if i != len(dets_f)-1:
        str_f += ","
plt.plot(x, dets_y, label="DETS, F=" + str_f + ", ep=" + str(ep))

plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.set_yscale('log')
ax.set_xlabel('Generation', fontname='Times New Roman')
ax.set_ylabel('Value', fontname='Times New Roman')
ax.set_title(function.__name__ + ' C=' + str(c), fontname='Times New Roman')
plt.savefig("fig/dets/" + function.__name__ + "_epsilon" + str(ep) + "_" + str(grp_num) + ".pdf")
plt.show()