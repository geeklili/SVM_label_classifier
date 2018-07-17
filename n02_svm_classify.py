import re
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC


def create_train_data():
    """构建训练集特征向量numpy_out_li: [[0,1,0,..],[1,0,0,..],[...],...]，
       以及标签向量feature_li: ['C01','C01','C02',...]
    """
    jieba.load_userdict('./data/user_dict.txt')
    with open('./data/job_classification.yml', 'r', encoding='utf-8') as f1:
        with open('./data/bag_of_words.txt', 'r', encoding='utf-8') as f2:
            pure_di = eval(f2.read())
            vec_length = len(pure_di)
            # print(vec_length)
            numpy_out_li = list()
            feature_li = list()
            for i in f1:
                line = i.strip().lower().split(': ')[0]
                class_line = i.strip().lower().split(': ')[1]
                line_li = list(jieba.cut_for_search(line))
                feature_li.append(class_line)

                line_str = ' '.join(line_li)
                line_str2 = re.sub(r'…|/|\(|\)|\.|）|（', '', line_str)
                line_lis = line_str2.split(' ')
                line_li_new = list()
                for j in line_lis:
                    if len(j) > 0:
                        line_li_new.append(j)

                # print(line_li_new)
                # numpy_in_li: [170, 104, 305, 221] -->数字代表第几个位置的向量的数字应该为1
                numpy_in_li = list()
                for k in line_li_new:
                    val = pure_di.get(k, None)
                    numpy_in_li.append(val)

                in_vec_li = list()
                print(numpy_in_li)
                for num in range(1, vec_length + 1):

                    if num in numpy_in_li:
                        in_vec_li.append(1)
                        # print(num)
                    else:
                        in_vec_li.append(0)
                numpy_out_li.append(in_vec_li)

    print(numpy_out_li)
    print(feature_li)
    return numpy_out_li, feature_li


def fit_train_data(train_li, feat_li):
    """
    训练模型，保存模型，查看模型准确率
    :param train_li: 特征值
    :param feat_li:标签值
    :return:
    """
    # print(train_li)
    # print(len(train_li))
    # print(len(feat_li))
    # print(feat_li)

    model = SVC(kernel='linear', C=100)
    model.fit(X=train_li, y=feat_li)
    joblib.dump(model, "./data/model/train_model.model")

    # 检验结果的准确率：通过预测训练集
    # y_pred = model.predict(train_li)
    # print('多输出多分类器预测输出分类:\n', y_pred)
    # n = 0
    # for i in range(len(feat_li)):
    #     if feat_li[i] == y_pred[i]:
    #         n += 1
    # print(n)
    # print(float(n / len(feat_li)))


def profession_predict(name):
    """预测岗位的职能
       1.构建岗位的特征向量
       2.预测岗位的职能
    """
    jieba.load_userdict('./data/user_dict.txt')
    with open('./data/bag_of_words.txt', 'r', encoding='utf-8') as f2:
        pure_di = eval(f2.read())
        vec_length = len(pure_di)
        # print(vec_length)
        li = list(jieba.cut_for_search(name))
        # print(li)
        line_str = ' '.join(li)
        line_str2 = re.sub(r'…|/|\(|\)|\.|）|（', '', line_str)
        line_lis = line_str2.split(' ')
        line_li_new = list()
        for j in line_lis:
            if len(j) > 0:
                line_li_new.append(j)
        # print(line_li_new)
        predict_li = list()

        numpy_in_li = list()
        for k in line_li_new:
            val = pure_di.get(k, None)
            numpy_in_li.append(val)
        # print(numpy_in_li)
        in_vec_li = list()
        for num in range(1, vec_length + 1):

            if num in numpy_in_li:
                in_vec_li.append(1)
                # print(num)
            else:
                in_vec_li.append(0)
        predict_li.append(in_vec_li)
        # print(predict_li)

    model = joblib.load("./data/model/train_model.model")
    predicted = model.predict(predict_li)

    return predicted


if __name__ == '__main__':
    train, features = create_train_data()

    fit_train_data(train, features)

    ret = profession_predict('人事经理')
    print(ret)
