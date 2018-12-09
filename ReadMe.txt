#author 	YSH
#version 	1.0 2017/04/25
#copyright	YSH 僅供學習交流

#Code執行說明
#Python使用套件PIL(讀取影像)numpy(陣列運算)matplotlib.pyplot(畫圖)matplotlib.mlab(PCA)
#以下必須在相同路徑下(main.py)(Data_Train資料夾)(Demo資料夾)

1.使用Database of Faces (AT&T Laboratories Cambridge)
Reference : http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html

2.開啟main.py

----------------main.py----------------
conductPCA=0
usetoolPCA=1
plotPCA=0
CreateTrainData=0
Train_1LayerModel=0
Valid_1LayerModel=0
Train_2LayerModel=0
Valid_2LayerModel=0
plotNN1=0
creatplotNN2data=0
----------------main.py----------------

3.以上為執行參數，執行該功能請設定值為1，關閉為0，功能依序為:

conductPCA        > 進行Data PCA並輸出降維度後的陣列在CSV檔案
usetoolPCA        > PCA是否使用tool box，否則使用bmpPCA.py進行
plotPCA           > 在CSV中PCA降維度陣列繪圖
CreateTrainData   > 用PCA創造隨機排序標記好的training Data  (對應函式內可修改總訓練資料量)
Train_1LayerModel > 訓練neural network model with 1-hidden layer (對應函式內可修learning rate)
Valid_1LayerModel > 產生結果驗證neural network model with 1-hidden layer (對應函式內可修改總驗證資料量)
Train_2LayerModel > 訓練neural network model with 2-hidden layer (對應函式內可修learning rate)
Valid_2LayerModel > 產生結果驗證neural network model with 2-hidden layer (對應函式內可修改總驗證資料量)
plotNN1           > 畫出neural network model with 1-hidden layer線性簡化分界
creatplotNN2data  > 利用整個座標產生輸出結果，請用matlab或excel查看分界 (把所有可能輸出會跑一段時間)

4.NeuralNet1.py 內可直接修改1-hidden layer參數初始隨機值和預設training Data總次數
5.NeuralNet2.py 內可直接修改2-hidden layer參數初始隨機值和預設training Data總次數
