# TTA 인증용 코드 사용법

[toc]

## 0. 요구사항

- docker
- nvidia-docker
- gpu 1ea 

## 1. 사전 준비

파일을 다운로드 후 압축을 해제한 뒤 `cd` 명령어를 이용해 directory 변경

```
unzip tta.zip -d tta
cd tta
```

### `/tta` 디렉토리내 파일 설명

```
/tta/
├── Conformer-CTC-BPE-8000-KOR.nemo
├── infer.py						
├── requirements.txt				
├── testset/
└── testset.csv
```

- `Conformer-CTC-BPE-8000-KOR.nemo` : model file 

- `infer.py` : python inference 용 파일

- `requirements.txt` : python package file list

- `testset/` : [AI Hub 한국어 자유대화 음성(일반남녀)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=109) 데이터의 Validation set 중 스튜디오 녹음 본 파일들

- `testset.csv` : testset의 label과 파일 명을 작성해둔`.csv`파일 

  

## 2. Docker 환경 구축

- Pull the docker container image
```bash
docker pull nvcr.io/nvidia/nemo:22.05
```
- Run the container 

```bash
docker run --rm --name asr_tta --gpus 0 --shm-size=64gb -v `pwd`:/tta -it nvcr.io/nvidia/nemo:22.05 /bin/bash
```

- move `/tta` dir

```bash
cd /tta
```



## 3. python package 설치

```bash
pip install -r requirements.txt
```



## 4. Inference

```
$ pwd 
/tta
$ python infer.py
```

### `infer.py` arguments 

```bash
usage: infer.py [-h] [--batch_size BATCH_SIZE] [--num_wokers NUM_WOKERS] [--write_result WRITE_RESULT]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        batch size for inference(default : 32)
  --num_wokers NUM_WOKERS
                        num of workers for inference dataloader(default : 4)
  --write_result WRITE_RESULT
                        result write option(default : True)
```



## 5. inference 결과

```
Transcribing: 100%|███████████████████████████████████████████| 846/846 [01:12<00:00, 11.68it/s]
result is saved ./result.csv
--------------------------------------------------------------------------------
|                     Accuracy : 91.6641% (CER is 8.3359%)                     |
--------------------------------------------------------------------------------
```

- inference가 배치사이즈 크기대로 진행되어 모든 파일에 대한 inference
- infercne가 종료 후  result.csv파일에 파일이름, reference(ground truth), prediction 저장
- CER을 계산하여 출력
- CER 계산 방법은 다음과 같음
```math
  CER = \frac{S+D+I}{N}
```
  - CER(Character Error Rate)

  - Substitution (S): 추론된 텍스트 중 정답 텍스트와 비교해 잘못 대체된 음절 수
  - Deletion (D): 추론된 텍스트 중 정답 텍스트와 비교해 잘못 삭제된 음절 수
  - Insertion (I): 추론된 텍스트 중 정답 텍스트와 비교해 잘못 추가된 음절 수
  - N: 정답 텍스트의 음절 수
