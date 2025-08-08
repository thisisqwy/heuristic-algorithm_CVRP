import random
import math
import numpy as np
import time

start_time = time.time()
class ACO_SDVRP():
    def __init__(self,data,):
        self.num_city=len(data)#表示所有点的数量，包括配送中心
        self.dist_matrix=self.compute_dis_mat(data)
        self.demands=[0,468, 335, 1, 170, 225, 479, 359, 463, 465, 206, 146, 282, 328, 462, 492]#客户需求量列表
        self.Q=500#每辆车的最大装载量
        self.alpha = 2  # 信息素重要程度因子
        self.beta = 3  # 启发函数重要因子 #启发函数就是点之间距离的倒数
        self.rho = 0.7  # 信息素持久因子
        self.C=1#常量系数，作为信息素常数
        self.K=20#蚂蚁数量
        self.NC=1000#迭代次数
        self.Tau = np.ones([self.num_city, self.num_city])  # 信息素矩阵,初始各点间的信息素设为1。
        self.Eta = 1/ self.dist_matrix  # 启发式函数
    #根据二维坐标计算各点间距离矩阵
    def compute_dis_mat(self,data):
        matrix=np.zeros((self.num_city,self.num_city))
        for i in range(self.num_city):
            for j in range(self.num_city):
                if i==j:
                    matrix[i,j]=np.inf#np.inf的倒数是0
                else:
                    matrix[i,j]=math.sqrt((data[i][0]-data[j][0])**2+(data[i][1]-data[j][1])**2)
        return matrix
    #规则公式，具体看论文
    def fx(self,current,x):
        return self.dist_matrix[current][x]+self.dist_matrix[x][0]
    #生成蚁群
    def get_ants(self):
        pop=[]
        start=0
        for i in range(self.K):
            routes=[]
            current = start
            route=[current]
            load=self.Q#load是每辆车的剩余运载能力
            demands=np.array(self.demands)#demands表示所有客户剩余需求量
            allowed=np.where((demands > 0) & (demands <= load))[0]#可允许访问城市对应的索引，最小索引至少为1
            while  allowed is not None:
                score=[]#score相当于概率公式中的分子
                # 通过信息素计算城市之间的转移概率
                for v in allowed:
                    score.append((self.Tau[current][v] ** self.alpha) * (self.Eta[current][v] ** self.beta))
                score_sum = sum(score)
                p = [x / score_sum for x in score]
                # 轮盘赌选择一个城市
                index = self.rand_choose(p)

                current = allowed[index]
                route.append(current)
                load=load-demands[current]
                demands[current]=0
                if not np.any(demands> 0):#如果每个客户需求都为0，车辆立刻从current返回配送中心
                    route.append(start)
                    routes.append(route.copy())
                    allowed=None
                else:#当有客户的需求未得到满足
                    if load == 0:#如果此时车辆已经没能力了，就立马返回中心并换一辆车。
                        route.append(start)
                        routes.append(route.copy())
                        load=self.Q
                        current=start
                        route=[current]
                        allowed=np.where((demands > 0) & (demands <= load))[0]
                    else:#此时load>0,而demands存在大于0的数，但demands中的数可能存在除一个大于load外全等于0的情况
                        if not np.any((demands > 0) & (demands <= load)):
                            allowed = np.where(demands > 0)[0]
                            func_values=[]
                            for x in allowed:
                                func_values.append(self.fx(current,x))
                            current = allowed[np.argmin(func_values)]
                            route.extend([current,start])#把剩下一点运载能力用完即返回中心
                            routes.append(route.copy())
                            demands[current]=demands[current]-load
                            load=self.Q
                            current=start
                            route=[current]
                            allowed = np.where((demands > 0) & (demands <= load))[0]
                        else:#如果存在任何可以被当前车辆完全服务的客户
                            allowed = np.where((demands > 0) & (demands <= load))[0]
            pop.append(routes.copy())
        return pop
    #计算单个蚂蚁代表的路径的长度
    def path_length(self,routes):
        total_dist = 0
        # 计算所有车辆路径长度之和
        for route in routes:#routes是一个大的numpy数组，里面是多个numpy数组表示每辆车的路径,格式为[0,x1,x2,0]
            for i in range(len(route) - 1):
                total_dist +=self.dist_matrix[route[i]][route[i + 1]]
        return total_dist
    #计算蚁群的适应度函数
    def fitness(self):
        scores=[]
        for ant in self.ants:
            length=self.path_length(ant)
            scores.append(1/length)
        return scores
    #轮盘赌选择
    def rand_choose(self, p):#轮盘赌函数用来选择处于点i时，下一个点j选择哪个。
        rand = np.random.rand()
        for i, prob in enumerate(p):
            rand -= prob
            if rand < 0:
                return i  # 找到就直接返回
    #更新信息素
    def update_Tau(self):
        delta_tau = np.zeros([self.num_city, self.num_city])  # delta_tau是i点到j点路上所有蚂蚁对这条路信息素浓度的增加量之和。先让这个二维数组等于一个0数组。
        for k,routes in enumerate(self.ants):
            for route in routes:
                for i in range(len(route) - 1):
                    a = route[i]  # a是route这条路径的当前起点。
                    b = route[i+1]  # b是route这条路径中点a的下一个点。
                    delta_tau[a][b] = delta_tau[a][b] + self.C*self.fits[k] # 如果只更新一条边的信息素浓度，那就是把此问题作为有向图即非对称旅行商问题看待的。
                    delta_tau[b][a] = delta_tau[b][a] + self.C*self.fits[k]
        self.Tau = self.rho * self.Tau + delta_tau
    #主函数
    def main(self):
        best_length = math.inf
        best_path = None
        for i in range(self.NC):
            # 生成新的蚁群，生成一次新的蚁群实际就是让蚁群重走一遍觅食路径。
            self.ants=self.get_ants()
            self.fits=self.fitness()
            # 取该蚁群的最优解
            tmp_length=1/max(self.fits)
            tmp_path=self.ants[np.argmax(self.fits)]
            # 更新最优解
            if tmp_length < best_length:
                best_length =tmp_length
                best_path = tmp_path
            # 更新信息素
            self.update_Tau()
            print(f"当前为第{i+1}代，历史最短路径长度为：{best_length:.2f}")
        print(f"最终的最短路径长度为：{best_length:.2f}")
        print(f"最终的最优路径为:{best_path}")

coordinates = [
            [0, 0],  # 配送中心
            [32, 41],  # 客户1
            [96, 9],  # 客户2
            [7, 58],  # 客户3
            [97, 87],  # 客户4
            [26, 21],  # 客户5
            [23, 100],  # 客户6
            [52, 31],  # 客户7
            [76, 43],  # 客户8
            [74, 17],  # 客户9
            [72, 104],  # 客户10
            [40, 99],  # 客户11
            [8, 16],  # 客户12
            [27, 38],  # 客户13
            [78, 69],  # 客户14
            [46, 16]  # 客户15
]
aco=ACO_SDVRP(coordinates)
aco.main()