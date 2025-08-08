import random
import numpy as np
import time
import math

start_time = time.time()
class GA_CVRP():
    def __init__(self,d,q):
        self.dist_matrix=d#各点间的距离矩阵
        self.demands=q#客户需求量
        self.cross_rate=0.9#交叉率
        self.mutate_rate=0.09#变异率
        self.num_client=8
        self.max_gen =25 #遗传算法迭代次数
        self.pop_size=20
        self.num_vehicles=2# 车辆总数
        self.vehicle_capacity=8#车辆载重限制
        self.max_distance=50# 车辆一次行驶的最大距离
        self.pw=100
        self.pop=self.init_population()
    #种群初始化，也是编码的过程
    def init_population(self):
        pop=[]
        base = list(range(1, self.num_client+ 1))
        for _ in range(self.num_client):
            random.shuffle(base)  # 随机打乱客户访问序
            pop.append(np.array(base))
        return pop
    #计算每个个体即每条染色体的适应度函数
    def fitness(self,chromosome):
        routes = self.decode(chromosome)#routes是一个列表，里面是两个数组，格式为[0,x1,x2,0]
        total_dist = 0
        # 计算所有车辆路径长度之和
        for route in routes:
            for i in range(len(route) - 1):
                total_dist +=self.dist_matrix[route[i]][route[i + 1]]
        # 若使用车辆数超限，则加大惩罚
        if len(routes) > self.num_vehicles:
            total_dist +=self.pw* (len(routes) -self.num_vehicles)
        return 1 / (total_dist)  # 适应度：距离越小适应度越高
    #对每条染色体进行解码，得到其表示的路径信息
    def decode(self,chromosome):
        routes=[]
        current_route=[0]  # 当前车辆路径起点为depot（0）
        load = 0  # 当前车辆已装载重量
        dist = 0  # 当前车辆已行驶距离
        last = 0  # 当前车辆上一次访问点
        for gene in chromosome:
            # 计算如果将gene加入当前车辆，是否超载或超距
            new_load = load +self.demands[gene]
            new_dist = dist +self.dist_matrix[last][gene] +self.dist_matrix[gene][0]
            if new_load <= self.vehicle_capacity and new_dist <=self.max_distance:
                # 若可行，则将gene代表的客户加入当前车辆服务的对象中
                current_route.append(gene)
                load = new_load
                dist +=self.dist_matrix[last][gene]
                last = gene
            else:
                # 否则，当前车辆返回depot并开启新车辆
                current_route.append(0)
                routes.append(np.array(current_route))#这里用array是因为实际是对current_coute进行了一个拷贝，不然后面current_route重置了，routes里的也会被改变。
                # 重置新车辆参数
                current_route = [0, gene]
                load=self.demands[gene]
                dist=self.dist_matrix[0][gene]
                last=gene
        # 终止时让最后一辆车返回depot
        current_route.append(0)
        routes.append(np.array(current_route))
        return routes
    #轮盘赌选择个体作为交叉和变异的父本
    def select(self,pop, fits):
        total_fit = sum(fits)
        p = [i / total_fit for i in fits]
        rand= np.random.rand()
        for i, sub in enumerate(p):
            if rand >= 0:
                rand -= sub
                if rand < 0:
                    index = i
        return pop[index].copy()
    # 交叉操作：部分匹配交叉（PMX）
    def crossover(self,p1, p2):
        size =self.num_client
        # 随机选两点作为交叉片段
        start = random.randint(0, size-1)
        end = random.randint(start + 1, size)
        child1,child2=[-1]*size, [-1]*size
        child1[start:end+ 1]=p2[start:end+ 1]
        child2[start:end+ 1]=p1[start:end+ 1]
        for i in range(size):
            if child1[i] == -1:
                gene1 = p1[i]
                while gene1 in child1:
                    gene1 = p1[np.where(p2 == gene1)[0][0]]
                child1[i] = gene1
            if child2[i] == -1:
                gene2 = p2[i]
                while gene2 in child2:
                    gene2 = p2[np.where(p1 == gene2)[0][0]]
                child2[i] = gene2
        return np.array(child1), np.array(child2)
    #变异，交换两点基因
    def mutate(self,chrom):#不需要return chrom，因为array是动态数据类型
        m1, m2 = sorted(random.sample(range(self.num_client), 2))#因为有sorted，所以随机生成的两个不同数字肯定是m1<m2
        chrom[m1], chrom[m2] = chrom[m2], chrom[m1]  # 交换两个基因
    #主函数
    def main(self):
        fits = [self.fitness(ch) for ch in self.pop]
        best_chrom, best_fit =self.pop[np.argmax(fits)].copy(), max(fits)
        for _ in range(self.max_gen):
            elite = self.pop[np.argmax(fits)].copy()
            new_pop=[elite]
            # 生成剩余的 pop_size - 1 个体
            while len(new_pop) < self.pop_size:
                p1 = self.select(self.pop, fits)
                p2 = self.select(self.pop, fits)
                if random.random() < self.cross_rate:
                    c1, c2 = self.crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]
                if random.random() <self.mutate_rate:
                    self.mutate(c1)
                if random.random() <self.mutate_rate:
                    self.mutate(c2)
                # 只添加剩余数量这么多的个体，防止超出
                if len(new_pop) + 2 <= self.pop_size:
                    new_pop.extend([c1, c2])
                else:
                    new_pop.append(c1)  # 只差一个时只加一个就够了
            self.pop = new_pop.copy()
            fits = [self.fitness(ch) for ch in self.pop]
        best_chrom= self.pop[np.argmax(fits)].copy()
        best_routes=self.decode(best_chrom)
        best_dist=0
        for route in best_routes:
            for i in range(len(route) - 1):
                best_dist+=self.dist_matrix[route[i]][route[i + 1]]
        print('最优路径为:',best_routes)
        print('最短总路径长度',best_dist)
# 定义一个9 * 9的二维数组表示配送中心(编号为0)与8个客户之间，以及8个客户相互之间的距离d[i][j]
d=np.array([[0, 4, 6, 7.5, 9, 20, 10, 16, 8],  # 配送中心（编号为0）到8个客户送货点的距离
     [4, 0, 6.5, 4, 10, 5, 7.5, 11, 10],  # 第1个客户到配送中心和其他8个客户送货点的距离
     [6, 6.5, 0, 7.5, 10, 10, 7.5, 7.5, 7.5],  # 第2个客户到配送中心和其他8个客户送货点的距离
     [7.5, 4, 7.5, 0, 10, 5, 9, 9, 15],
     [9, 10, 10, 10, 0, 10, 7.5, 7.5, 10],
     [20, 5, 10, 5, 10, 0, 7, 9, 7.5],
     [10, 7.5, 7.5, 9, 7.5, 7, 0, 7, 10],
     [16, 11, 7.5, 9, 7.5, 9, 7, 0, 10],
     [8, 10, 7.5, 15, 10, 7.5, 10, 10, 0]])
# 8个客户分布需要的货物的需求量，第0位为配送中心自己
q= np.array([0,1, 2, 1, 2, 1, 4, 2, 2])
ga=GA_CVRP(d,q)
ga.main()
end_time = time.time()
print("代码运行时间：", end_time - start_time, "秒")
