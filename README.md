## 训练

### 安装
```bash
pip3 install -r requirements.txt
```

### 数据集
701_StillsRaw_full.zip 下载链接：
http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip

LabeledApproved_full 下载链接：
http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/data/LabeledApproved_full.zip

label_colors 链接：
http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/data/label_colors.txt

注意：label_colors 最后有一个空行

### 运行
#### 1.生成数据集：
```python
python3 python/CamVid_utils.py
```

#### 1.生成数据集：
3. 训练生成模型：
python3 python/train.py CamVid

4. 推理代码：
python3 python/inference.py
- default dataset is CamVid

create a directory named "CamVid", and put data into it, then run python codes:
```python
python3 python/CamVid_utils.py 
python3 python/train.py CamVid
```

- or train with CityScapes

create a directory named "CityScapes", and put data into it, then run python codes:
```python
python3 python/CityScapes_utils.py 
python3 python/train.py CityScapes
```

