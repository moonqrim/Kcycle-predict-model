import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

# real : 출주표 데이터 처리한 변수
# grade : 경기 등급 ( 1:선발 / 2:우수 / 3:특선 )

# eXpected Prize
class Xp():
    def __init__(self):
        self.model = None
        self.grade = None
        self.input, self.target = None, None
        self.train_input, self.train_target, self.test_input, self.test_target \
            = None, None, None, None

    # 등급 별 모델 학습
    def model_(self, grade):
        # kpi.csv : 전처리가 끝난 데이터 파일
        df = pd.read_csv('kpi.csv')
        df = df[df['AVRG_GTSR_TMS3_VALUE'] > 0]
        df['AVG_RATIO'] = (df['VICTRY_RT'] + df['YUNDAE_RT'] + df['THREE_YUNDAE_RT']) / 3

        g1 = df[df['PLAYER_GRAD_NM'] == '특선']
        g2 = df[df['PLAYER_GRAD_NM'] == '우수']
        g3 = df[df['PLAYER_GRAD_NM'] == '선발']

        t_col1 = ['AVG_RATIO', 'AVG_RANK', 'GRADE_UP', 'GRADE_DOWN', 'PRE_WIN_PER_GAME', 'EXCD_WIN_PER_GAME',
                  'OVRTK_WIN_PER_GAME', 'DEFNS_WIN_PER_GAME', 'AVRG_GTSR_TMS3_VALUE', 'IN_3rd']

        if grade == 1:
            df1 = g3[t_col1].dropna(axis=0)
        elif grade == 2:
            df1 = g2[t_col1].dropna(axis=0)
        elif grade == 3:
            df1 = g1[t_col1].dropna(axis=0)

        self.grade = grade

        train = df1.iloc[:, :-1]
        target = df1['IN_3rd']

        df_input = train.astype(float).to_numpy()
        df_target = target.astype(int).to_numpy()

        train_input, test_input, train_target, test_target = \
            train_test_split(df_input, df_target, random_state=42, test_size=0.2)

        self.train_input = train_input
        self.train_target = train_target
        self.test_input = test_input
        self.test_target = test_target

        logi = LogisticRegression(C=1, random_state=42)
        logi.fit(train_input, train_target)

        self.model = logi
        self.input, self.target = df_input, df_target

    # 선수 별 확률 출력 및 시각화
    def vis(self, real):
        real_input = np.array(real).astype(float)
        proba2 = self.model.predict_proba(real)

        cnt, race_num = 1, 1
        win_lst, lose_lst, back_num = [], [], ['1p', '2p', '3p', '4p', '5p', '6p', '7p']
        for r, p in zip(real_input, proba2): # iterrows
            if cnt == 1:
                print(f'({race_num})번 경주')
                print()

            lose_lst.append(p[0]) # 모델이 거짓으로 예측한 확률을 담은 리스트
            win_lst.append(p[1]) # 모델이 참으로 예측한 확률을 담은 리스트

            print(f'{cnt}번 확률 : {(p[1] * 100).round(3)} %')

            # 경주 당 선수 7명 전제
            if cnt == 7:
                if self.grade == 1:
                    winp = plt.bar(back_num, win_lst, color='lightgrey', width=0.5)
                    losep = plt.bar(back_num, lose_lst, color='dodgerblue', bottom=win_lst, alpha=0.15, width=0.5)
                elif self.grade == 2:
                    winp = plt.bar(back_num, win_lst, color='lightseagreen', width=0.5)
                    losep = plt.bar(back_num, lose_lst, color='dodgerblue', bottom=win_lst, alpha=0.15, width=0.5)
                elif self.grade == 3:
                    winp = plt.bar(back_num, win_lst, color='crimson', width=0.5)
                    losep = plt.bar(back_num, lose_lst, color='dodgerblue', bottom=win_lst, alpha=0.15, width=0.5)

                plt.title(f'({race_num}) race')
                plt.xlabel('player')
                plt.ylabel('Probability in 3rd')
                plt.legend((winp, losep),('prize', 'fail'), fontsize=10)
                plt.show()
                race_num += 1
                print('==============================================')
                print()
                cnt, win_lst, lose_lst = 0, [], []

            cnt += 1

    # 모델 성능 테스트
    def model_test(self):
        y_pred = self.model.predict(self.train_input)

        #  precision : (양성으로 예측한 샘플 중 실제로 양성인 샘플의 수) / (양성으로 예측한 샘플의 총 수)
        precision_score_ = precision_score(self.train_target, y_pred)
        #  recall : (양성으로 예측한 샘플 중 실제로 양성인 샘플의 수) / (실제 양성인 샘플의 총 수)
        recall_score_ = recall_score(self.train_target, y_pred)
        f1 = f1_score(self.train_target, y_pred, average='macro')

        print(f'precision : {precision_score_}')
        print(f'recall : {recall_score_}')
        print(f'fi score : {f1}')

if __name__ == '__main__':
    # 10/20 광명 우수 (샘플)
    # 출주표 정보를 처리한 변수
    real = [
        [19.5, 3.833333333, 0, 0, 0, 0, 0.288461538, 0.326923077, 91.99],
        [3, 5.666666667, 0, 0, 0.022222222, 0, 0.066666667, 0.066666667, 90.39],
        [5, 6, 0, 0, 0, 0, 0, 0.208333333, 89.94],
        [55.5, 2.5, 0, 0, 0.085106383, 0.212765957, 0.255319149, 0.255319149, 94.38],
        [3, 4.5, 0, 0, 0, 0, 0.04, 0.1, 91.38],
        [31, 2.8, 0, 0, 0, 0, 0.358974359, 0.435897436, 92.66],
        [15.5, 3.666666667, 0, 0, 0.044444444, 0.044444444, 0.066666667, 0.311111111, 92.02],

        [41, 3, 0, 0, 0.04, 0.12, 0.28, 0.24, 92.38],
        [27.5, 1.833333333, 0, 100, 0.021276596, 0.063829787, 0.29787234, 0.127659574, 94.69],
        [39, 3, 0, 0, 0, 0.039215686, 0.392156863, 0.294117647, 93.5],
        [22.5, 5, 0, 0, 0.0625, 0.145833333, 0.145833333, 0.145833333, 91.39],
        [41.5, 5.333333333, -100, 0, 0, 0, 0.384615385, 0.173076923, 90.01],
        [5, 5.833333333, 0, 0, 0.115384615, 0, 0.019230769, 0.076923077, 90.24],
        [16.5, 4.2, 0, 0, 0.225, 0.05, 0.075, 0.1, 91.29],

        [9.5, 4.5, 0, 0, 0.042553191, 0.021276596, 0.106382979, 0.127659574, 91.14],
        [28, 3.25, 0, 0, 0.023255814, 0.255813953, 0.11627907, 0.209302326, 91.89],
        [53, 4.75, 0, 0, 0.591836735, 0.102040816, 0.020408163, 0.020408163, 92.39],
        [35.5, 2.4, 0, 0, 0, 0, 0.3125, 0.375, 93.21],
        [25, 2.333333333, 0, 100, 0.020833333, 0.020833333, 0.25, 0.083333333, 94.63],
        [2, 5.2, 0, 0, 0, 0.02173913, 0.108695652, 0.173913043, 90.91],
        [20, 5.6, 0, 0, 0.058823529, 0.058823529, 0.215686275, 0.156862745, 90.7],

        [5, 3.5, 0, 0, 0.083333333, 0, 0.020833333, 0.104166667, 91.46],
        [21, 3.166666667, 0, 100, 0, 0.041666667, 0.145833333, 0.104166667, 93.66],
        [50, 5.666666667, -100, 0, 0.029411765, 0.176470588, 0.323529412, 0.088235294, 90.52],
        [52, 2.333333333, 0, 0, 0.111111111, 0.222222222, 0.355555556, 0.066666667, 94.56],
        [18, 5, 0, 0, 0, 0.021276596, 0.14893617, 0.404255319, 90.92],
        [11.5, 4.166666667, 0, 0, 0, 0.038461538, 0.076923077, 0.288461538, 91.11],
        [38.5, 3.5, 0, 0, 0.395833333, 0.020833333, 0.0625, 0.041666667, 91.92],

        [51, 2.25, 0, 0, 0.4, 0.244444444, 0, 0.155555556, 93.28],
        [4.5, 4.5, 0, 0, 0, 0, 0.022727273, 0.204545455, 90.98],
        [0, 5.166666667, 0, 0, 0, 0, 0, 0.125, 89.96],
        [29.5, 4.5, 0, 0, 0.152173913, 0.195652174, 0.108695652, 0.130434783, 91.97],
        [33, 6.333333333, 0, 0, 0.283018868, 0.113207547, 0.150943396, 0.037735849, 90.99],
        [38.5, 2.166666667, 0, 0, 0, 0.0625, 0.291666667, 0.270833333, 93.77],
        [14.5, 5, 0, 0, 0.193548387, 0.129032258, 0.032258065, 0.096774194, 91.79],

        [32.5, 3.4, 0, 0, 0.088235294, 0.088235294, 0.117647059, 0.352941176, 91.46],
        [15.5, 3.8, 0, 0, 0.125, 0.041666667, 0.104166667, 0.1875, 91.47],
        [67.5, 1, 0, 0, 0.5, 0.239130435, 0.130434783, 0, 95.97],
        [38, 4, 0, 0, 0.044444444, 0.066666667, 0.355555556, 0.288888889, 92.7],
        [5.5, 4.666666667, 0, 0, 0, 0, 0.042553191, 0.212765957, 90.88],
        [5, 5.333333333, 0, 0, 0.039215686, 0.058823529, 0.058823529, 0.078431373, 90.01],
        [3, 4.666666667, 0, 0, 0, 0.020833333, 0.083333333, 0.083333333, 91.17]
    ]

    xP = Xp()
    xP.model_(grade=2)  # 등급을 선택하여 모델 학습
    xP.vis(real=real)  # 출주표 정보를 바탕으로 확률 산출 및 시각화
    #xP.model_test()  # 모델 성능 테스트