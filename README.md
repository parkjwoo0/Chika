# 3조 Rhythm-of-Chika


## 1. 프로젝트의 목적 및 용도.
어린이들이 양치질을 꺼려하는 상황 속에서 부모들이 아이들에게 양치질 교육을 하는데 어려움이 있다.

구글의 오픈소스인 ‘mediapipe’를 활용해 양치하는 모습의 손 관절 데이터를 수집하고, 파이토치를 이용하여 양치 부위를 예측하는 모델을 만들고자 했다. 

양치 부위 예측 모델을 게임에 적용하여 어릴 때부터 양치 습관을 올바르게 길러주는 에듀테인먼트 컨텐츠를 제작함으로써 양치질에 대한 아이들의 거부감을 해소할 수 있다.

효과 : 값비싼 전동칫솔을 구매하지 않고 카메라만으로 모든 구역을 구분할 수 있어 사용자들로 하여금 각 양치 부위를 정확히 그리고 꼼꼼히 닦도록 도와줄 수 있다.


## 2. 프로젝트 시작 방법 및 코드 설명
requirements.txt 

```
pip install -r requirements.txt
``` 

필요 라이브러리는 requirements.txt를 통해 install가능하다.

 
 ```
 python collector_final.py
 ```

### 데이터 수집 방법 

collector_final.py를 실행시켜 400프레임 동안 (약 13초) 해당 양치 부위를 닦는다. 각 부위마다 numpy 배열 한 개씩 생성된다. 그리고 pause time(5초) 동안 다음 부위로 이동해 또 양치할 준비를 한다. 그렇게 해서 총 16개 부위를 닦고 배열들을 저장해서 총 100개의 세션을 쌓았다. 


### 데이터 전처리

collector를 통해 21개 랜드마크에 대한 x,y,z 상대좌표와 가속도가 각 부위마다 400프레임씩 쌓인 상태이다. 하나의 세션은 (400, 3) 크기의 numpy 위치 배열과 (398, 3) 크기의 가속도 배열 16개씩으로 구성되었다고 볼 수 있다. 본 프로젝트에서 피쳐로 선정한 칫솔의 기울기를 구하고 이상치 변환을 위해 다음과 같은 함수를 정의했다. 

- hyperparameter: 각 부위마다 연결했을 때 칫솔의 기울기와 가장 일치하는 기울기를 지니는 두 점을 각각 gradlab1, gradlab2로 지정했고, 랜드마크의 가속도끼리 비교할 때 상관계수가 낮은 두 점을 acclab1과 acclab2로 지정했다.  
- distance: 두 점의 기울기를 구해주는 함수
- coefficient: 혹시라도 데이터를 쌓는 과정에서 다음 부위로 넘어갈 때 잘못 움직이는 경우가 있을 수 있으니, 가장 운동성이 일정한 가운데 300프레임으로 잘라준다. IQR 방식으로 이상치를 찾고 본 데이터의 특성이 시계열 데이터임을 고려해 이상치 데이터를 양옆 데이터의 평균으로 변환한다. 또한 가장 첫번째 또는 마지막이 이상치일 상황 또한 대처했다. 
- acceleration: 가속도 또한 coefficient 함수와 마찬가지로 IQR 방식의 이상치 변환을 진행했다. 또한 mediapipe에서 얻어온 z값은 손목을 기준으로 랜드마크들의 깊이값을 가져오는데 양치할 때마다 기준이 되는 손목이 자꾸 움직이기 때문에 활용하기 어렵다고 판단하여 z값을 제거한 x,y 가속도 값 2개만 return하도록 한다. 
- preprocessing: 정답 데이터셋을 구성하는 함수다. 해당 부위에 대한 기울기, 가속도 x,y값 2쌍, 그리고 정답 레이블 ‘1’을 붙인 (300, 6) 크기의 배열을 num_session(100개)만큼 반복하면서 밑에 계속 붙여주며 정답 데이터셋을 만들어준다. 
- preprocessing2: 오류 데이터셋을 구성하는 함수다. 정답 부위를 제거한 gesture_list에서 랜덤으로 돌아가며 오답 부위들의 기울기, 가속도 x,y값 2쌍, ‘0’ 레이블을 쌓는다. 오류 데이터 개수는 num_session2로 지정해준다.  

```
def preprocessing(gesture):             # 정답데이터 셋 구성 
    data2 = np.zeros((1,300,6))
    for i in range(1, num_session+1):
        data = np.zeros((300,1))
        pos = np.load('%s_%s_p_%d.npy' % (hand, gesture, i))  # 위치데이터 로드
        acc = np.load('%s_%s_a_%d.npy' % (hand, gesture, i))  # 가속도데이터 로드
        acc1 = Acceleration(pos, acc, acclab1)    # 첫 번재 랜드마크 가속도 데이터       
        acc2 = Acceleration(pos, acc, acclab2)    # 두 번째 랜드마크 가속도 데이터
        coef = Coefficient(pos)       # 기울기 데이터 (랜드마크 이은)
        if gesture == right_gesture:
            label = np.ones((300,1))
        else:
            label = np.zeros((300,1))
        data = np.hstack((data,acc1,acc2,coef,label))
        data = data[:, 1:].reshape((1,300,6))
        data2 = np.vstack((data2,data))
        
    return data2[1:]
    
```

### 모델 소개


```
class LSTM_Chicka(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, dropout_rate):
        super(LSTM_Chicka,self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers  #layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.dropout_rate = dropout_rate  #dropout_rate
        self.lstm =  nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, dropout = dropout_rate, batch_first = True)
        self.layer_out = nn.Linear(hidden_size, num_classes) 
        self.sigmoid = nn.Sigmoid()  #이중분류 시 sigmoid 함수 활용
        
    def forward(self,x):
        out, _status = self.lstm(x)
        out = self.layer_out(out[:, -1]) 
        out = self.sigmoid(out)

        return out    
 ```
 
 
시계열 모델인 LSTM을 사용하였고 이진분류이기 때문에 num_classes는 1로 지정하였습니다.
LSTM 계층을 num_layers = 2로 두개 쌓았습니다. 이진분류이기 때문에 활성화함수로 sigmoid함수를 사용하였습니다. 
many to one 방식의 양치구역을 구별했기에 16개의 양치구역 구별 모델을 만들었습니다.
 
  ```
  criterion = torch.nn.BCELoss()     #이중분류시 사용하는 binarycrossentropy 손실함수 사용 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam 제일 성능이 좋았음
  
   ```
   
   optimizer로는 Adam이 가장 좋은 성능을 보여 Adam을 사용하였으며, 이진 분류로 Binary CrossEntropy를 loss function으로 선택하였습니다.
   
   
 ### validation test시에 나온 결과입니다(DMF)
   
   ![image](https://user-images.githubusercontent.com/74550931/186811601-b363417f-0cab-443b-a287-bda7eba68edd.png)



   
   
   ## 3. 외부 리소스 정보
mediapipe hand solution을 참조해 손 관절 21개의 랜드마크 좌표 정보를 얻어옴. 

https://google.github.io/mediapipe/solutions/hands.html 

https://github.com/google/mediapipe 

LSTM 모델 관련 코드 참조

https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py.
