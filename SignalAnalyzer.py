#%%
# 신호 기록 가져오기
with open('sample_20200601_pointfinger.txt', 'r') as openfile :
    samples = openfile.readlines()

tmp_timests = [ samples[i][:-1] for i in range(len(samples)) if i%3==0 ]
tmp_samples = [ samples[i][:-1] for i in range(len(samples)) if i%3==1 ]

#%%
# 중복된 시간 기록 제거
timests, samples = list(), list()
deleted = list()
for sinx in range(len(tmp_timests)-1) :
    if tmp_timests[sinx] != tmp_timests[sinx+1] :
        samples.append(float(tmp_samples[sinx]))
        timests.append(float(tmp_timests[sinx].replace('2020-06-01 09:', '')[3:]))
        if tmp_timests[sinx].replace('2020-06-01 09:', '')[:2] == '26' : timests[-1] += 60

#%%
# 플롭 해보기
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
fm.get_fontconfig_fonts()
matplotlib.rc('font', family=fm.FontProperties(fname='C:/Windows/Fonts/NanumSquarel.ttf').get_name())
def plot(t, s, title='근전도 신호 데이터', xlabel='시간(초)', ylabel='신호 세기', style='-') :
    T = np.array(t)
    Y = np.array(s)
    mat = np.array([T, Y])

    plt.figure(figsize=(18, 5))
    plt.plot(T, Y, style, ms=15, lw=1)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontsize=15, pad=20)
    plt.show()

plot(timests, samples)

#%%
# 초기화 시간 제거하기
init_end_inx = 0
while True :
    if samples[init_end_inx] < 190 :
        break
    init_end_inx += 1
timests, samples = timests[init_end_inx:], samples[init_end_inx:]
plot(timests[init_end_inx:], samples[init_end_inx:])

# # 기울기 값 계산
# grads = list()
# for i in range(1, len(timests)) :
#     grads.append((samples[i-1]-samples[i])/(timests[i-1]-timests[i]))

# plot(timests[1:], np.abs(grads))

#%%
# 바이어스 하기
bias_value = 173.5
plot(timests, np.abs(np.array(samples)-bias_value))


#%%
#구간 구하기
timespan = [51, 53.5] # 중간 신호
timespan = [53.5, 56.5] # 없는 신호

timespan = [75, 77] # 짧은 신호
timespan = [62, 66] # 긴 신호
span_indice = [0, 0]
while span_indice[0] < len(timests) :
    if timests[span_indice[0]] > timespan[0] : break
    span_indice[0] += 1
while span_indice[1] < len(timests) :
    if timests[span_indice[1]] > timespan[1] : break
    span_indice[1] += 1
span_indice

#%%
# 푸리에 주기적 곱
bias_samples = abs(np.array(samples)-bias_value)[span_indice[0]:span_indice[1]]
low_samples, high_samples = np.copy(bias_samples), np.copy(bias_samples)
pass_filter_value = 4
high_samples[bias_samples <= pass_filter_value] = 0
low_samples[bias_samples > pass_filter_value] = 0
plt.figure(figsize=(18, 8))
plt.plot(np.convolve(low_samples, high_samples), '-', ms=1, lw=1)

# plt.xlabel("")
# plt.ylabel("")
# plt.title(title, fontsize=15, pad=20)
plt.show()

#%%
bias_samples = abs(np.array(samples)-bias_value)
low_samples, high_samples = np.copy(bias_samples), np.copy(bias_samples)
high_samples[bias_samples <= pass_filter_value] = 0
low_samples[bias_samples > pass_filter_value] = 0
step_num = 5
pack_num = 30
pinx = 0
pvalues = list()
while pinx+pack_num < len(samples) :
    pvalues.append(np.sum(np.convolve(
        low_samples[pinx:pinx+pack_num], 
        high_samples[pinx:pinx+pack_num]))/pack_num)
    pinx += step_num
plot(timests, samples)
plot(timests[:len(pvalues)], pvalues)

#%%
plot(timests[:len(pvalues)], pvalues)

#%%
# 그룹화하기
ginx = -1
group_area = list()
while ginx < len(pvalues) :
    ginx += 1
    while ginx < len(pvalues) and pvalues[ginx] < 1:
        ginx += 1
    tmp = ginx
    while ginx < len(pvalues) and pvalues[ginx] > 0:
        ginx += 1
    group_area.append((tmp, ginx))
group_area

#%%
# 최댓값 보기
maximums = list()
for area in group_area :
    if pvalues[area[0]:area[1]] :
        maximums.append(np.max(pvalues[area[0]:area[1]]))
plot([i for i in range(len(group_area)-1)], maximums, style='.')

#%%
(timests[0] - timests[1]) * 50

#%%
# CNN의 입력 행렬 만들기
# 이상점 제거
group_area = group_area[1:-1]

#%%
# 훈련 데이터 만들기
labels = []
for i in range(19) :
    labels.append(i)
len(labels)

#%%
def make_train(groups, areas, max_depth=40) :
    train_list = list()
    for area in areas :
        train_set = list()
        label_set = list()
        group_span = (area[1] - area[0])
        # if group_span : continue
        inx = 0
        while inx < max_depth and inx < group_span :
            a = labels[int(inx/group_span*len(labels))] # angle
            v = groups[inx:group_span]+[0 for i in range(max_depth-(group_span-inx))]
            train_set.insert(0, v)
            label_set.insert(0, a)
            inx += 1
        if train_set :
            train_list.append((train_set, label_set))
    return train_list

trains = make_train(pvalues, group_area)

#%%
tmp_train_inputs = list(map(lambda l: l[0], trains))
tmp_label_inputs = list(map(lambda l: l[1], trains))
train_inputs = list()
label_inputs = list()
for i in range(len(tmp_train_inputs)) :
    for j in range(len(tmp_train_inputs[i])) :
        train_inputs.append(tmp_train_inputs[i][j])
        label_inputs.append(tmp_label_inputs[i][j])
train_inputs = np.array(train_inputs)
label_inputs = np.array(label_inputs)
print(train_inputs.shape)

#%%
### 신경망 학습을 위한 모듈 가져오기
import tensorflow as tf
from tensorflow.keras import datasets, layers, Sequential

#%%
# 모델 만들기
model = Sequential()
model.add(layers.Dense(40, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(40, activation='relu'))
model.add(layers.Dense(30, activation='relu'))
model.add(layers.Dense(len(labels), activation='softmax'))
model.build(input_shape=(None, 40))
model.summary()

#%%
# 훈련 시작
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_inputs, label_inputs, epochs=500)

#%%
# 모델 평가
test_loss, test_acc = model.evaluate(train_inputs,  label_inputs, verbose=2)

#%%
predicts = list()
for i in range(len(train_inputs)) :    
    p = model.predict(train_inputs[i:i+1]).tolist()[0]
    predicts.append(p)

#%%
predicts = [ p.index(max(p)) for p in predicts]
predicts

#%%
plot([i for i in range(len(predicts))], predicts, xlabel="필터링된 그룹", ylabel="카테고리", title="예측 데이터")
plot([i for i in range(len(predicts))], label_inputs, xlabel="필터링된 그룹", ylabel="카테고리", title="검증 데이터")

#%%
model.save('./my_model.h5')
#%%
print(train_inputs[0:1])
#%%
# len(pvalues)

# #%%
# # 스펙트럼 관찰
# import scipy.signal

# f, P = scipy.signal.periodogram(np.array(samples), int(1/(timests[1]-timests[0])), nfft=len(samples))

# plt.subplot(211)
# plt.plot(f, P)
# plt.title("선형 스케일")

# plt.subplot(212)
# plt.semilogy(f, P)
# plt.title("로그 스케일")

# plt.tight_layout()
# plt.show()
