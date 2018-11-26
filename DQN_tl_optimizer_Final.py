import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
import csv
import json
from subprocess import call
from xml.etree.ElementTree import ElementTree, parse, dump

# 교통신호최적화 알고리즘의 DQN 에이전트
class DQNAgent :

    def __init__(self, actions):

        # 행동 세팅
        self.actions = actions

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.batch_size = 3
        self.train_start = 30

        # 리플레이 메모리 (최대 크기 = 2000)
        self.memory = deque(maxlen=2000)

        # 모델과 타겟 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타겟 모델 초기화
        self.update_target_model()

    # 상태가 입력, Q함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=1, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(2, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    # 타겟 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # 현재 상태(교통량) 추출 : SALT 시뮬레이터 초기 결과 값
    def get_state(self, ):
        count = 0
        avg_veh_num = 0

        with open('output/test-1-node-PeriodicOutput.csv') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                count += 1
                avg_veh_num += float(row['avgvehnum'])

        avg_veh_num = avg_veh_num / count

        return avg_veh_num

    # 현재 상태 기반 행동 선택 : 앱실론-그리디 정책 기반 (랜덤하게 선택 또는 Q테이블 기반 선택)
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(2)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # 선택된 행동 기반 신호체계 수정 : 신호체계(tss.xml)에서 Duration 값 증가 또는 감소하도록 수정.
    def action_policy(self, action, epoch):

        # 첫 Epoch 에는 초기 신호체계 파일을 읽고 다음 Epoch 때부터는 이전 신호체계 파일을 읽음.
        if epoch == 1:
            previous_tll_file = 'data/test-1-node/tss.xml'
        else:
            previous_tll_file = 'tlls/tll_epoch-' + str(epoch - 1) + '.xml'
        tree = parse(previous_tll_file)
        root = tree.getroot()

        # 선택된 행동 기반 신호체계 수정 : [0]:duration 증가 / [1]: duration 감소
        if action == 0:
            for i in range(5, 10):
                root[0][i][0].set("duration", str(int(root[0][i][0].get("duration")) + 1))
        else:
            for i in range(5, 10):
                if int(root[0][i][0].get("duration")) - 20 > 0:
                    root[0][i][0].set("duration", str(int(root[0][i][0].get("duration")) - 1))

        # 수정된 신호체계 파일 저장.
        output_tll_file = 'tlls/tll_epoch-' + str(epoch) + '.xml'
        tree.write(output_tll_file)

    # SALT 시뮬레이터 실행 : 현재 [상태, 행동] 기반 다음 상태 및 보상 추출
    def run_salt(self, state, action, epoch):

        reward = 0

        # SALT 시뮬레이터 설정 파일 경로.
        config_file = 'data/test-1-node/scenario.json'

        # SALT 시뮬레이터 설정 파일 수정 : 매번 Epoch 마다 이전 신호체계 파일을 읽도록 설정 파일을 수정.
        with open(config_file) as r:
            data = json.load(r)
            if epoch != 1:
                data["input"]["trafficLightSystem"] = '../../tlls/tll_epoch-' + str(epoch - 1) + '.xml'
            else:
                data["input"]["trafficLightSystem"] = 'tss.xml'
            print('\n\n\nDATA --------------------------------------------------------------------------------------\n',
                  data["input"]["trafficLightSystem"])

        with open(config_file, 'w') as w:
            json.dump(data, w, ensure_ascii=False)

        # SALT 시뮬레이터 실행
        call(['./saltalone', config_file])

        # 다음 상태 추출
        next_state = round(agent.get_state(), 2)

        # 현재 상태와 다음 상태를 비교하여 보상 값 설정
        if state < next_state:
            reward = -1
        else:
            reward = 1

        return next_state, reward

    # 리플레이 메모리에 샘플 <state, action, reward, next_state'> 저장
    def append_sample(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):

        # 앱실론 값 감소하도록 조정
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 리플레이 메모리에서 배치 크기 만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, 1))
        next_states = np.zeros((self.batch_size, 1))
        actions, rewards = [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]

        # 현재 상태에 대한 모델의 Q함수
        # 다음 상태에 대한 타겟 모델의 Q함수
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # 벨만 방정식을 이용한 타겟 모델 업데이트
        for i in range(self.batch_size):
            if rewards[i] == 1:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)


if __name__ == "__main__":

    reward_list = []
    state_list = []
    action_list = []
    total_action = 0
    epoch = 0
    total_reward = 0

    min_state = 10
    j = 0

    # 액션 정의 - [0]:Duration 증가 / [1]:Duration 감소
    action = [0,1]

    # DQN 에이전트 생성
    agent = DQNAgent(action)

    # Epoch 500 번까지 강화학습 실행
    for e in range(0,500):
        epoch=epoch+1

        # 현재 상태(교통량) 추출 : SALT 시뮬레이터의 초기 값 (output/test-1-node-PeriodicOutput.csv)
        original_state = round(agent.get_state(), 2)
        state = np.reshape(original_state,[1,1])

        # 현재 상태 기반 행동 선택 : 0 또는 1
        action = agent.get_action(state)

        # 선택된 행동 기반 신호 체계 수정 : [0]:Duration+1 / [1]:Duration-1
        agent.action_policy(action, epoch)

        # 새로운 상태 및 행동 기반 다음 상태 및 보상 추출 : SALT 시뮬레이터 실행
        next_state, reward = agent.run_salt(state, action, epoch)

        # 리플레이 메모리에 샘플 <state, action, reward, next_state'> 저장
        agent.append_sample(state, action, reward, next_state)

        # 리플레이 메모리 기반 강화학습 훈련
        if len(agent.memory) >= agent.train_start:
            agent.train_model()

        # 타겟 모델을 모델의 가중치로 업데이트
        if reward == 1:
            agent.update_target_model()

        total_reward+=reward
        total_action+=action
        reward_list.append(reward)
        state_list.append(original_state)
        action_list.append(action)

        print("\nSTATE LIST = ",state_list)
        #print("\nREWARD LIST = ",reward_list)
        print("\nTOTAL REWARD = ",total_reward)
        #print("\nACTION LIST = ",action_list)

        # print("\nTOTAL ACTION = ",total_action)

        print("\nEPOCH : ", epoch, "  STATE : ", state, "  memory length:",
              len(agent.memory), "  epsilon:", agent.epsilon)
        print("\nMIN STATE : ", min_state)

        # 학습 중단 조건 : 최소 교통량이 5번 연속 발생하면 학습 중단
        if min_state > original_state:
            min_state = original_state

        elif min_state == original_state:
            j += 1
            if j == 4:
                break
        else:
            j = 0


