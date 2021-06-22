# MineRL_Distributed-Imitation-Learning
# Index

1. **Overview**
2. **Implementation Spec**
3. **Imitation Learning Method (DQfD)**
4. **Distributed Learning Method (Ape-X)**
5. **Experiments**
6. **Results**
7. **Conclusion**

# 1. Overview

 본 프로젝트는 모방학습과 분산처리 기술을 적용하고 마인크래프트 에이전트의 성능을 향상시키는 것을 목표로합니다. 본 프로젝트 필요한 것들은 다음과 같습니다.

- MineRL Environment & Network Architecture
- Data for Imitation Learning
- Ray
- PER
- DQFD
- Ape-x

## **MineRL Treechop Env & Network Architecture**

 마인크래프트 강화학습 환경으로 나무를 캐면 보상을 얻는 환경입니다. 환경에서 64x64x3의 RGB 이미지 데이터를 에이전트에게 전달합니다.  이미지를 에이전트의 관측값(Observation)으로 사용되고, 뉴럴네트워크를 통해 출력되는 값을 Action index로 사용합니다. 따라서 Action discretization을 통해 Action space를 분리하고 각 인덱스에 해당하는 적절한 행동을 정의해서 mapping 해주어야 합니다. 

## **Data for Imiation Learning**

 카네기 멜론 대학교의 MinerlLabs에서 Sample Efficient Learning을 위한 데이터들을 제공합니다. 그 중에서도 1.5GB에 해당하는 MineRL Treechop 데이터만을 사용합니다. 데이터는 그림과 같이 metadata, recording.mp4, rendered.npz 파일들로 이루어져 있습니다. recoding의 경우 사람이 녹화한 기록이며 rendered에는 동영상의 프레임마다 사람이 취한 행동들을 numpy 배열로 만든 파일입니다. 

## **Ray**

Ray는 분산처리 환경을 쉽게 구현할 수 있도록 해주는 파이썬 라이브러리입니다. ray를 이용해서 각 스레드별로 프로세스를 할당할 수 있으며, 프로세스 간 데이터 공유, 비동기처리를 지원합니다. 

전체적인 과정은 위 그림의 두 단계로 진행됩니다. 위 그림의 Step 1이 DQFD에 해당하고 Step 2가 Ape-X Framework에 해당합니다. DQFD를 통해서 에이전트 네트워크를 사전학습을 진행합니다. 그 다음으로 사전학습된 네트워크로 Learner와 Actor의 네트워크를 초기화 한 후, 분산 강화학습이 진행됩니다. 

## Prioritized Experience Replay (PER)

 기존의 Experience Replay 방식은 리플레이 버퍼에 단순하게 트랜지션(상태, 행동, 보상, 다음 상태, 에피소드 정보)을 채워넣어서 랜덤하게 추출함으로써 버퍼에 쌓여 있는 모든 데이터를 학습에 활용하게 됩니다. 하지만 이 방법은 불필요한 데이터도 활용하기 때문에 데이터마다 우선순위를 부여한 Prioritized Experience Replay가 연구되었습니다. PER에서는 i번째 트랜지션 데이터가 뽑힐 확률을 다음과 같이 정의합니다. 힙 구조를 통해 구현되며 우선순위는 TD Error 방식, Rank Based 방식 두 가지로 부여하게 됩니다. 본 프로젝트에서는 TD Error 방식을 사용합니다. 

$$P(i) = {p_i^\alpha \over \sum_k p_k^\alpha}, ~~~ p_i = |\delta_i| +\epsilon  ~~ or ~ {1 \over rank(i)}$$

## Deep Q Learning from Demonstrations (DQFD)

DQFD는 기존의 Deep Q Learning에 사용되는 Experience Replay에 전문가의 데이터를 넣어서 학습하는 방식입니다. 기존의 DQN은 epsilon을 기반으로 무작위 탐색을 통해 쌓은 데이터를 uniform sampling 방식으로 학습했습니다. DQFD의 경우는 사람의 데이터를 쓰기 때문에 더 효율적으로 에이전트를 원하는 방향으로 학습할 수 있습니다. 기존의 Q loss에 margin classification loss가 더해진 손실함수를 사용하게 됩니다. 

$$J_E(Q) = \max_{a \in A}[Q(s,a) + l(a_E, a)] - Q(s, a_E)$$

## Distributed Prioritized Experience Replay (Ape-X)

Ape-X Framework는 기존의 DQN 알고리즘을  Actor와 Learner로 분리하고 공유된 Prioritized Experience Replay를 사용하여 학습효율성을 높인 학습방식입니다. Actor의 경우 분산처리를 통해 구현되며 여러개의 환경인스턴스를 동시에 실행하여 트랜지션 샘플링 속도를 빠르게 합니다.  

이제 본격적으로 알고리즘의 학습방식과 구현을 살펴보겠습니다.

# 2. Implementation Spec

본 프로젝트를 위해서는 다음과 같은 사항들이 구현되어야 합니다

Step 1 : DQFD

- DQN (model)
- Prioritized Experience Replay (replay buffer)
- train_dqn method
- margin_loss method
- pre_train method

Step 2 : Ape-X

- Actor Class
- Learner Class
- Visualization module

# 3. **Imitation Learning Method (DQfD)**

 DQfD에서 가장 핵심적인 기능은 1) 트랜지션 데이터를 배치 단위 로드 2) 데이터 전처리 (action mapping) 3) 리플레이 버퍼에 추가 4) 모델 학습 세 가지로 나뉘게 됩니다. 

1) 데이터 로드
위에서 기술한 minerl 데이터들은 (64x64x3) RGB raw 픽셀값들을 가지는 이미지이며, 이를 배치 단위, 시퀀스 단위로 로드하게 된다면 (batch_size, sequence_length, 64, 64, 3) 크기의 5차원 텐서를 로드하게 됩니다.

2) 데이터 전처리
다음으로 5차원 텐서의 각 이미지들에 대해 action mapping을 수행합니다. 에이전트의 행동의 자유도를 높이기 위해 총 19개의 인덱스를 사용하였습니다.

3) 리플레이 버퍼 추가
action mapping을 수행한 데이터들을 TD Error를 계산한 후 PER에 추가합니다.

4) 모델 학습
이제 모델을 학습합니다. 먼저 리플레이 버퍼에서 샘플링을 진행한 후, 트랜지션의 각 정보들(state, action, next_state, reward, done)에 대해 배치를 구성합니다. 다음으로 배치의 트랜지션들에 대해 q_value 및 next_q_vqlue를 계산해서 q_loss를 계산한 후, 역전파를 통해 모델을 학습시킵니다.

# 4. **Distributed Learning Method (Ape-X)**

Ape-X는 Framework는 기존의 DQN에서 Actor와 Learner를 분리한 것이 핵심이며 논문에서 제시된 그림과 같이 크게 세 가지 부분으로 구성되어 있습니다. 

1) Actor Class 
Ape-X에서 Actor는 환경의 인스턴스를 독립적으로 실행해서 샘플링을 담당합니다. 

Learner의 네트워크를 가져와서 실제 환경에서 행동을 수행합니다. 

Actor들은 각각 스레드를 할당받아서 서로 개별적으로 샘플링을 수행하고 공유된 리플레이 버퍼에 트랜지션을 추가합니다.

핵심메소드로는 update_parmas와 explore가 있고 각각 actor network의 업데이트와 환경에서 탐색을 수행합니다.

2) Learner Class 
Ape-X에서 네트워크의 업데이트를 담당하는 클래스입니다. 공유된 리플레이 버퍼로부터 샘플링을 진행하여 배치를 구성하고 그래디언트를 계산하여 네트워크 업데이트를 진행합니다. 네트워크 업데이트 후에는 TD error에 따른 우선순위를 다시 계산해서 리플레이 버퍼에 반영합니다.

핵심 메소드로는 update_network, update_target_network가 있습니다.

3) Shared Replay Buffer
off-policy reinforcement learning을 가능하게 해주는 핵심적인 요소입니다. 에이전트가 환경에서 수행한 행동들과 그에 대한 트랜지션 데이터들을 저장해서 학습을 효율적으로 진행하게 해줍니다.

