from typing_extensions import get_args
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

class DE:
    def __init__(self, n, d, f, c, function, iter, g):
        """
        n:個体数
        d:次元数
        f:突然変異係数
        c:交叉率
        function:目的関数
        iter:試行回数
        g:世代数
        """
        self.n = n
        self.d = d
        self.f = f
        self.c = c
        self.iter = iter
        self.g = g
        self.x_max = function("max")
        self.x_min = function("min")
        self.function = function

    def init_population(self):
        self.population = np.random.rand(self.n, self.d) * (self.x_max - self.x_min) + self.x_min

# 定義域から外れた時の処理
    def domain_check(self, v, base_x):
        for i in range(self.d):
            if v[i] <= self.x_min:
                v[i] = base_x[i] + np.random.rand() * (self.x_min - base_x[i])
            elif v[i] >= self.x_max:
                v[i] = base_x[i] + np.random.rand() * (self.x_max - base_x[i])
        return v

    def mutation(self, current_x_idx):
        """
        current_x_idx:ターゲットベクトルのインデックス
        """
        num = np.arange(self.n) # 解候補の何番目をとるか決めるための、0~N-1の数が入った配列を作成
        num = np.delete(num, current_x_idx) # ターゲットベクトルの番号は削除
        selected_num = np.random.choice(num, 3, replace = False) #ターゲットベクトル以外から重複なしで３つの解候補を抽出
        base_x = self.population[selected_num[0]]
        v = base_x + self.f * (self.population[selected_num[1]] - self.population[selected_num[2]]) # vの計算
        return self.domain_check(v, base_x)

    def crossover(self, current_x, v):
        """
        current_x:ターゲットベクトル
        v:変異ベクトル
        """
        z = np.zeros(self.d) # 交叉ベクトル(解候補になるかもしれないベクトル)
        for i in range(self.d):
            n_1 = np.random.rand() #0~1の実数をランダムで生成
            if n_1 <= self.c: #条件に当てはまれば、zの要素にvの要素を代入
                z[i] = copy.deepcopy(v[i])
            else:
                z[i] = copy.deepcopy(current_x[i]) #条件を満たさなければ、xの要素をそのまま代入
        return z

    def get_best_x(self, current_x, z):
        if self.function(current_x) >= self.function(z):
            return z
        else:
            return current_x

    def get_best_solution(self):
        solution = []
        for i in range(self.n):
            solution.append(self.function(self.population[i]))
        return min(solution)

# 世代を更新する
    def update_population(self):
        next_population = []
        for idx, current_x in enumerate(self.population):
            v = self.mutation(idx)
            z = self.crossover(current_x, v)
            next_population.append(self.get_best_x(current_x, z))
        self.population = next_population

    def optimize(self):
        self.init_population()
        for _ in range(self.g):
            self.update_population()
        return self.get_best_solution()

# グラフを作る場合(最良解のリストを作る)
    def get_best_solution_list(self):
        self.init_population()
        best_solution_list = []
        for _ in range(self.g):
            self.update_population()
            best_solution_list.append(self.get_best_solution())
        return np.array(best_solution_list)

# 最良解の平均値を求める
    def get_average_best_solution_list(self):
        average_best_solution_list = np.zeros(self.g)
        for _ in tqdm(range(self.iter)):
            average_best_solution_list += self.get_best_solution_list()
        return average_best_solution_list / self.iter

# グラフを作る
    def get_graph(self):
        x = np.arange(self.g)
        y = self.get_average_best_solution_list()
        plt.plot(x, y)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)
        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.set_yscale('log')
        ax.set_xlabel('Generation', fontname='Times New Roman')
        ax.set_ylabel('Value', fontname='Times New Roman')
        plt.savefig("fig/de/sphere.pdf")
        plt.show()
