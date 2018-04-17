# LSTM_Classify_text
# LSTM_TextClassify_word2vec 基于word2vector将每一个词转换为固定大小的input_size,采用双层LSTM进行训练，分类结果百分之九十左右.使用google训练好的模型GoogleNews-vectors-negative300.bin.gz， 模型链接: https://pan.baidu.com/s/1lnFJYrOkzE17tBe5Q4RfaQ 密码: dce2 采用数据 aclImdb 处理流程： 1.首先将要分类的文本数据分词（中文需分词，英文不用） 2.将每个词转换成固定大小的vector，并保存至文件中，以备训练加载 3.将以上数据送入双层LSTM训练即可。 4.根据训练好的模型进行分类
5.里面的路径等参数需根据自己的情况进行修改
