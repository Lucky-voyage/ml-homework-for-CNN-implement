# <center>Digit Recognition Task</center>
<div style="text-align: center;">
    <div style="display: inline-block; text-align: left;">
        姓名：王臣龙<br>
        学号：U202215504<br>
        班级：CS2205
    </div>
</div>

### 文件结构<br>
· `experiment`:该文件包含一个main()函数，该函数有且仅有参数`train`，设置为`True`时
读取`data/train.csv`并进行推理任务。该文件组织所有的类和方法，完成整个预测流程。<br><br>
· `evaluate`:该文件加载参数`params/model.json`并调用`utils.Evaluate`来进行评估任务。
终端将会显示四项评估结果，并显示混淆矩阵。<br><br>
· `train`: 该文件组织所有类和方法，实现模型训练操作。<br><br>
· `requirements`: 介绍运行环境。<br><br>
· `config.yml`: 地址等参数列表。<br><br>
· `palyground`: 本项目notebook文件。<br><br>

| Folder     | Function      | Files                                                                  |
|------------|---------------|------------------------------------------------------------------------|
| layers     | 各神经层的实现       | AvgPooling、Conv2d、DepthWise、Linear、Relu                                | 
| utils      | 模型构建以及训练的基本单元 | data_loader、evaluate、get_checkpoint、loss_function、model、Transpose_Conv | 
| data       | 存放训练集和测试集     | test.csv、train.csv                                                     | 
| outputs    | csv、png等输出文件  | ...                                                                    | 
| params     | 模型参数加载        | model.json、model.pt                                                    |  
| other_mode | 存放其他模型的文件夹    | SVM、DescisionTree                                                      | 



