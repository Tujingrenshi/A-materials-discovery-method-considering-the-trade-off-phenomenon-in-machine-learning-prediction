import numpy as np

class MidPoint():
    """Class for selecting the midpoints for a candidate for the threshold."""
    def __init__(self):
        pass

    def __call__(self, start, end, seed):
        return (start + end) / 2


class NormalGaussianDistribution():
    """Class for stochastically determining candidate for the threshold using 
    a normal distribution.
    """
    def __init__(self, n_sigma):
        self.n_sigma = n_sigma


    def __call__(self, start, end, seed):
        if seed is not None:
            np.random.seed(seed)
        # 生成正态分布的随机数
        candidate = np.random.normal(
            loc=(start + end) / 2,
            scale=(end - start) / (2 * self.n_sigma)
        )

        i = 0  # gai
        while candidate < start or candidate > end:  # gai
            np.random.seed(i)  # gai
            candidate = np.random.normal(  # gai
                loc=(start + end) / 2,  # gai
                scale=(end - start) / (2 * self.n_sigma)  # gai
            )  # gai
            i += 1  # gai

        # if candidate < start:
        #     print("start")
        #     candidate =  (start + end) / 2
        # if candidate > end:
        #     print("end")
        #     candidate = (start + end) / 2
        # candidate = np.clip(candidate, start, end)  # gai
        # 确保生成的数在start和end之间
        return candidate

if __name__ == '__main__':
    vals = [-5, -4, -3, -2, -1, 0, 1, 2]
    #
    # selector = MidPoint()
    # thresholds = []
    # start = vals[0]
    # for end in vals[1:]:
    #     thresholds.append(selector(start, end, None))
    #     start = end
    # print(thresholds)
    #
    # selector = NormalGaussianDistribution(1)
    # thresholds = []
    # start = vals[0]
    # for end in vals[1:]:
    #     thresholds.append(selector(start, end, seed=None))
    #     start = end
    # print(thresholds)
    #
    # selector = NormalGaussianDistribution(5)
    # thresholds = []
    # start = vals[0]
    # for end in vals[1:]:
    #     thresholds.append(selector(start, end, seed=None))
    #     start = end
    # print(thresholds)
    #
    #
    #
    # candidate = -5
    # start = -3
    # end = -1
    # result = np.cilp(candidate, start, end)
    # print(result)

    selector = NormalGaussianDistribution(1)
    thresholds = []
    start = -5
    end = -1
    for seed in range(100):
        threshold = selector(start, end, seed=None)
    if threshold < start and threshold > end:
        print(threshold)

